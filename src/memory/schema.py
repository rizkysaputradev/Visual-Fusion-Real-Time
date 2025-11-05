# src/memory/schema.py
"""
Vision-Fusion-RT — Memory Schema & Label Registry
-------------------------------------------------

Purpose
- Define the canonical structures for dynamic, online memory:
  * stable integer IDs for vectors (int64)
  * label ↔ ids mapping
  * per-vector metadata (timestamp, source, optional attrs)
- Provide a threadsafe LabelRegistry with CRUD operations and persistence.

Design
- Each vector stored in ANN backend is assigned a unique int64 ID.
- label → set(ids) mapping for quick retrieval, bookkeeping, eviction.
- id → VectorMeta metadata mapping for auditability and debugging.
- Persistence is JSON (human-readable) + optional gzip; index files are handled
  by the backend (e.g., FAISS writes its own binary).

Concurrency
- All mutating API calls acquire a lock; read-only calls are lock-free when safe.

You’ll typically use this registry from the FAISS store wrapper, e.g.:
    reg = LabelRegistry()
    ids = store.add(vecs, metas=[VectorMeta(label="mug") for _ in vecs], registry=reg)
    reg.save("data/registries/labels.json")

This module is backend-agnostic.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Iterable, Tuple, Any
import json
import gzip
import os
import threading
import time


# -------------------------
# Dataclasses
# -------------------------

@dataclass
class VectorMeta:
    """
    Per-vector metadata stored alongside the ANN index.
    """
    label: str
    ts: float = 0.0                    # wall/monotonic time when captured
    source: str = "rt"                 # e.g., "rt", "file", "script"
    attrs: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # attrs may be None → normalize to {}
        if d.get("attrs") is None:
            d["attrs"] = {}
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "VectorMeta":
        return VectorMeta(
            label=str(d.get("label", "")),
            ts=float(d.get("ts", 0.0)),
            source=str(d.get("source", "rt")),
            attrs=dict(d.get("attrs", {}) or {}),
        )


class LabelRegistry:
    """
    Thread-safe mapping between labels and vector IDs + per-ID metadata.

    Internal state
    --------------
    _label_to_ids : dict[str, set[int]]
    _id_to_meta   : dict[int, VectorMeta]
    _next_id      : int (monotonic increasing; never reused automatically)

    Notes
    -----
    - IDs are int64-compatible; Python int is fine.
    - Deletions remove IDs from both mappings; IDs are not recycled (to keep
      provenance clear). You *can* reset _next_id manually if desired.
    """

    def __init__(self):
        self._label_to_ids: Dict[str, set[int]] = {}
        self._id_to_meta: Dict[int, VectorMeta] = {}
        self._next_id: int = 1_000_000  # start high to avoid clashing with FAISS default 0..N-1
        self._lock = threading.Lock()

    # -------------- Introspection --------------

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            n_labels = len(self._label_to_ids)
            n_ids = len(self._id_to_meta)
            sizes = {lbl: len(s) for lbl, s in self._label_to_ids.items()}
            return {
                "n_labels": n_labels,
                "n_vectors": n_ids,
                "sizes": sizes,
                "next_id": self._next_id,
            }

    def labels(self) -> List[str]:
        with self._lock:
            return sorted(self._label_to_ids.keys())

    def ids_for(self, label: str) -> List[int]:
        with self._lock:
            return sorted(self._label_to_ids.get(label, set()))

    def meta_of(self, vid: int) -> Optional[VectorMeta]:
        return self._id_to_meta.get(int(vid))

    # -------------- Allocation --------------

    def allocate_ids(self, n: int) -> List[int]:
        """Allocate `n` fresh IDs (int) atomically."""
        with self._lock:
            start = self._next_id
            self._next_id += int(n)
            return list(range(start, start + n))

    # -------------- Mutations --------------

    def add(self, label: str, ids: Iterable[int], metas: Iterable[VectorMeta]) -> None:
        """
        Register IDs under a label and attach metadata per ID.
        """
        label = str(label)
        ids = [int(i) for i in ids]
        metas = list(metas)
        if len(ids) != len(metas):
            raise ValueError("ids and metas length mismatch")
        with self._lock:
            S = self._label_to_ids.setdefault(label, set())
            for vid, m in zip(ids, metas):
                S.add(vid)
                self._id_to_meta[vid] = m

    def remove_ids(self, ids: Iterable[int]) -> None:
        """Remove specific IDs from the registry (all labels)."""
        ids = [int(i) for i in ids]
        with self._lock:
            for vid in ids:
                # remove from label buckets
                meta = self._id_to_meta.pop(vid, None)
                if meta is not None:
                    bucket = self._label_to_ids.get(meta.label)
                    if bucket is not None and vid in bucket:
                        bucket.remove(vid)
                        if not bucket:
                            # drop empty bucket
                            self._label_to_ids.pop(meta.label, None)

    def remove_label(self, label: str) -> List[int]:
        """Remove all IDs under a label. Returns the IDs removed."""
        with self._lock:
            ids = list(self._label_to_ids.pop(label, set()))
            for vid in ids:
                self._id_to_meta.pop(vid, None)
            return ids

    def rename_label(self, old: str, new: str) -> None:
        """Rename a label (keeps the same IDs)."""
        if old == new:
            return
        with self._lock:
            ids = self._label_to_ids.pop(old, None)
            if ids is None:
                return
            self._label_to_ids[new] = self._label_to_ids.get(new, set()).union(ids)
            # update metas
            for vid in ids:
                m = self._id_to_meta.get(vid)
                if m is not None:
                    m.label = new

    # -------------- Persistence --------------

    def save(self, path: str) -> None:
        """
        Save registry to JSON (gz if path ends with .gz). Format:
        {
          "next_id": int,
          "id_to_meta": { "123": {...}, ... },
          "label_to_ids": { "mug": [123, 456], ... }
        }
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._lock:
            data = {
                "next_id": self._next_id,
                "id_to_meta": {str(k): self._id_to_meta[k].to_dict() for k in self._id_to_meta},
                "label_to_ids": {lbl: sorted(list(ids)) for lbl, ids in self._label_to_ids.items()},
            }

        raw = json.dumps(data, ensure_ascii=False, indent=2)
        if path.endswith(".gz"):
            with gzip.open(path, "wb") as f:
                f.write(raw.encode("utf-8"))
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(raw)

    @staticmethod
    def load(path: str) -> "LabelRegistry":
        reg = LabelRegistry()
        if not os.path.exists(path):
            return reg
        if path.endswith(".gz"):
            with gzip.open(path, "rb") as f:
                raw = f.read().decode("utf-8")
        else:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
        obj = json.loads(raw)
        reg._next_id = int(obj.get("next_id", 1_000_000))

        id_to_meta = obj.get("id_to_meta", {}) or {}
        for k, v in id_to_meta.items():
            vid = int(k)
            reg._id_to_meta[vid] = VectorMeta.from_dict(v)

        label_to_ids = obj.get("label_to_ids", {}) or {}
        for lbl, ids in label_to_ids.items():
            reg._label_to_ids[str(lbl)] = set(int(i) for i in ids)

        return reg
