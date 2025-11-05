# src/memory/faiss_store.py
"""
Vision-Fusion-RT — FAISS Store Wrapper
--------------------------------------

Features
- Supports Flat (L2/IP), IVF (coarse quantizer), and IVF+PQ indices via a simple string spec:
    "Flat"                 -> IndexFlat (wrapped in IDMap)
    "IVF4096,Flat"         -> IVF (nlist=4096) with Flat residuals
    "IVF4096,PQ64"         -> IVF with Product Quantization (m=64)
- Inner-product or L2 distance (metric="ip" or "l2")
- ID-mapped add/remove/search (IndexIDMap2 wrapper so we can delete by IDs)
- Training logic for IVF/PQ (auto-train when needed)
- Thread-safe operations
- Persistence (write/read_index) + JSON sidecar for configuration
- Optional L2 normalization on add/query for cosine usage

Notes
- FAISS removal requires an IDMap wrapper; for Flat we wrap as well.
- IVF indexes must be trained before add(); wrapper auto-trains when `ntotal==0`
  or when an explicit `maybe_train()` is called and we have enough samples.
- Cosine similarity: when metric="ip", ensure vectors are L2-normalized upstream
  or set `enforce_normalize=True` to normalize here.

Example
    store = FaissStore(dim=512, metric="ip", index_spec="IVF4096,PQ64")
    ids = store.add(vecs, ids=[...])   # if ids None, registry.allocate_ids(...) first
    D, I = store.search(q, k=5)

    store.save("experiments/results/faiss_index.bin")
    store2 = FaissStore.load("experiments/results/faiss_index.bin")
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Tuple, Dict, Any
import json
import os
import threading

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise ImportError("faiss is required for FaissStore. Install faiss-cpu or faiss-gpu.") from e


# -------------------------
# Utilities
# -------------------------

_METRIC = {
    "ip": faiss.METRIC_INNER_PRODUCT,
    "l2": faiss.METRIC_L2,
}


def _ensure_float32(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    return a


def _l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


# -------------------------
# FaissStore
# -------------------------

@dataclass
class FaissConfig:
    dim: int
    metric: str = "ip"            # "ip" or "l2"
    index_spec: str = "Flat"      # "Flat", "IVF4096,Flat", "IVF4096,PQ64"
    nprobe: int = 16              # IVF probing
    enforce_normalize: bool = False  # Normalize on add/query when metric="ip"
    train_threshold: int = 200    # Train IVF/PQ after this many new vectors


class FaissStore:
    """
    High-level FAISS index with ID mapping and safe multithreaded access.
    """

    def __init__(self, cfg: FaissConfig):
        self.cfg = cfg
        self._dim = int(cfg.dim)
        if cfg.metric not in _METRIC:
            raise ValueError("metric must be 'ip' or 'l2'")
        self._metric = _METRIC[cfg.metric]
        self._index = self._build_index(cfg.index_spec)
        self._lock = threading.RLock()
        self._pending_to_train = 0  # count vectors added since last (re)train

    # -------------- Properties --------------

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def ntotal(self) -> int:
        with self._lock:
            return int(self._index.ntotal)

    @property
    def is_trained(self) -> bool:
        with self._lock:
            return bool(self._index.is_trained)

    # -------------- Build / Train --------------

    def _build_index(self, spec: str):
        """
        Parse index specification string and construct a FAISS index wrapped by IDMap2.
        """
        spec = spec.strip()
        if spec == "Flat":
            base = faiss.IndexFlat(self._dim, self._metric)
        else:
            # e.g., "IVF4096,Flat" or "IVF4096,PQ64"
            parts = [p.strip() for p in spec.split(",")]
            if len(parts) != 2 or not parts[0].startswith("IVF"):
                raise ValueError(f"Invalid index_spec: {spec}")
            nlist = int(parts[0][3:])
            if parts[1].lower() == "flat":
                quant = faiss.IndexFlat(self._dim, self._metric)
                base = faiss.IndexIVFFlat(quant, self._dim, nlist, self._metric)
            elif parts[1].upper().startswith("PQ"):
                # e.g., PQ64 → m=64, code size 8 by default (faiss chooses)
                m = int(parts[1][2:])
                quant = faiss.IndexFlat(self._dim, self._metric)
                base = faiss.IndexIVFPQ(quant, self._dim, nlist, m, 8, self._metric)
            else:
                raise ValueError(f"Unknown sub-spec: {parts[1]}")
        # IDMap2 for deletions and explicit IDs
        index = faiss.IndexIDMap2(base)
        if isinstance(base, (faiss.IndexIVFFlat, faiss.IndexIVFPQ)):
            base.nprobe = int(self.cfg.nprobe)
        return index

    def maybe_train(self, data: Optional[np.ndarray] = None) -> None:
        """
        Train IVF/PQ indexes when needed. For Flat, this is a no-op.
        If `data` is None, training will be deferred until `add()` sees data.
        """
        with self._lock:
            base = self._index.base_index if hasattr(self._index, "base_index") else self._index
            if isinstance(base, faiss.IndexIVF):
                if base.is_trained:
                    return
                if data is None or len(data) == 0:
                    return
                base.train(_ensure_float32(data))
                self._pending_to_train = 0

    # -------------- Add / Remove / Search --------------

    def add(self, vecs: np.ndarray, ids: np.ndarray | List[int]) -> List[int]:
        """
        Add vectors with explicit IDs. Returns list of IDs actually added.
        """
        vecs = _ensure_float32(vecs)
        ids = np.asarray(ids, dtype=np.int64)
        if vecs.ndim != 2 or vecs.shape[1] != self._dim:
            raise ValueError(f"vecs must be [N, {self._dim}] float32")
        if ids.shape[0] != vecs.shape[0]:
            raise ValueError("ids and vecs must have same length")

        if self.cfg.metric == "ip" and self.cfg.enforce_normalize:
            vecs = _l2_normalize(vecs)

        with self._lock:
            base = self._index.base_index if hasattr(self._index, "base_index") else self._index
            # Train IVF if needed
            if isinstance(base, faiss.IndexIVF) and (not base.is_trained):
                base.train(vecs)
                self._pending_to_train = 0
            self._index.add_with_ids(vecs, ids)
            self._pending_to_train += int(vecs.shape[0])
            return ids.tolist()

    def remove_ids(self, ids: Iterable[int]) -> int:
        """
        Remove the given IDs from the index. Returns number removed.
        """
        ids = np.asarray(list(ids), dtype=np.int64)
        with self._lock:
            # faiss ID selectors use Int64
            sel = faiss.IDSelectorArray(ids.size, faiss.swig_ptr(ids))
            removed = self._index.remove_ids(sel)
            # removed is number of vectors removed (int64)
            return int(removed)

    def search(self, q: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query the index.

        Returns
        -------
        (D, I)
            D: distances/similarities [N, k]
            I: ids [N, k]
        """
        q = _ensure_float32(q)
        if q.ndim == 1:
            q = q[None, :]
        if q.shape[1] != self._dim:
            raise ValueError(f"q must have dim={self._dim}")
        if self.cfg.metric == "ip" and self.cfg.enforce_normalize:
            q = _l2_normalize(q)

        with self._lock:
            D, I = self._index.search(q, int(k))
            return D.copy(), I.copy()

    # Convenience: single query
    def search1(self, q: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        return self.search(q, k=k)

    # -------------- Persistence --------------

    def save(self, index_path: str) -> None:
        """
        Save the FAISS index and JSON sidecar with config.
        Writes:
            <index_path>           — faiss binary
            <index_path>.json      — config (dim, metric, spec, nprobe, ...)
        """
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        with self._lock:
            faiss.write_index(self._index, index_path)
        cfg = asdict(self.cfg)
        with open(index_path + ".json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(index_path: str) -> "FaissStore":
        if not os.path.exists(index_path):
            raise FileNotFoundError(index_path)
        idx = faiss.read_index(index_path)
        # Restore config (if available); else infer minimal fields
        cfg_path = index_path + ".json"
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            cfg = FaissConfig(**obj)
        else:
            # Best-effort inference
            dim = int(idx.d)
            metric = "ip" if idx.metric_type == faiss.METRIC_INNER_PRODUCT else "l2"
            cfg = FaissConfig(dim=dim, metric=metric, index_spec="(loaded)", nprobe=16)
        store = FaissStore(cfg)
        # Replace internal index with loaded one (wrapped IDMap if not already)
        if not isinstance(idx, faiss.IndexIDMap2):
            idx = faiss.IndexIDMap2(idx)
        store._index = idx
        return store
