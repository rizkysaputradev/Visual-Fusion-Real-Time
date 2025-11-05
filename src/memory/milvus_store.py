# src/memory/milvus_store.py
"""
Vision-Fusion-RT — Milvus Store Wrapper (optional backend)
----------------------------------------------------------

This module provides a thin, *opinionated* wrapper over Milvus (via `pymilvus`)
to emulate the same interface as our FAISS store:

Required interface (subset parity with FaissStore):
- .dim : int
- .ntotal : int            (estimated via collection.num_entities)
- add(vecs, ids) -> List[int]
- remove_ids(ids) -> int
- search(q, k=5) -> (D, I) with numpy arrays of shapes [N, k]
- save(index_path) / load(index_path)   (no-op placeholders; Milvus is server-side)
- maybe_train(data=None)                (no-op; Milvus training handled by index creation)

Design
- We create a single collection named `{name}` with schema:
    id: INT64 (primary key, user-specified)
    emb: FLOAT_VECTOR(dim)
  Optional fields can be added later.
- Index is created on `emb` with provided metric ("IP" or "L2") and typ "IVF_FLAT" or "IVF_SQ8"/"HNSW".

Notes
- Milvus is networked; you must have a running Milvus instance.
- This wrapper prefers *sync* operations for simplicity.
- Persistence: `save()` writes a small JSON connection/config sidecar for reproducibility.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Tuple, Dict, Any
import json
import os
import numpy as np

try:
    from pymilvus import (
        connections, FieldSchema, CollectionSchema, DataType,
        Collection, utility,
    )
    _HAS_MILVUS = True
except Exception as e:
    _HAS_MILVUS = False
    _IMPORT_ERR = e


@dataclass
class MilvusConfig:
    dim: int
    metric: str = "ip"              # "ip" or "l2"
    host: str = "127.0.0.1"
    port: str = "19530"
    collection_name: str = "vision_fusion_rt"
    index_type: str = "IVF_FLAT"    # "IVF_FLAT" | "IVF_SQ8" | "HNSW"
    index_param: Dict[str, Any] = None   # e.g., {"nlist": 4096} or {"M": 16, "efConstruction": 200}
    search_param: Dict[str, Any] = None   # e.g., {"metric_type":"IP","params":{"nprobe":16}}

    def build_defaults(self):
        if self.index_param is None:
            self.index_param = {"nlist": 4096}
        if self.search_param is None:
            self.search_param = {"metric_type": "IP" if self.metric == "ip" else "L2", "params": {"nprobe": 16}}


class MilvusStore:
    """
    Minimal Milvus wrapper mirroring FaissStore semantics where reasonable.
    """

    def __init__(self, cfg: MilvusConfig):
        if not _HAS_MILVUS:
            raise ImportError("pymilvus is required for MilvusStore. pip install pymilvus") from _IMPORT_ERR
        self.cfg = cfg
        self.cfg.build_defaults()
        self._dim = int(cfg.dim)
        connections.connect("default", host=cfg.host, port=cfg.port)

        self._coll = self._ensure_collection(cfg.collection_name)
        # Create index if not exists
        self._ensure_index()

    # ---------------- Properties ----------------

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def ntotal(self) -> int:
        return int(self._coll.num_entities)

    # ---------------- Internals -----------------

    def _ensure_collection(self, name: str) -> "Collection":
        if utility.has_collection(name):
            coll = Collection(name)
            # Basic dim check (best-effort)
            return coll

        id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False)
        emb_field = FieldSchema(name="emb", dtype=DataType.FLOAT_VECTOR, dim=self._dim)
        schema = CollectionSchema(fields=[id_field, emb_field], description="Vision-Fusion-RT embeddings")
        coll = Collection(name=name, schema=schema)
        return coll

    def _ensure_index(self) -> None:
        # If there's no index, create it
        if len(self._coll.indexes) == 0:
            metric = "IP" if self.cfg.metric == "ip" else "L2"
            index = {
                "index_type": self.cfg.index_type,
                "metric_type": metric,
                "params": self.cfg.index_param or {},
            }
            self._coll.create_index(field_name="emb", index_params=index)
        # Load into memory for search
        self._coll.load()

    # ---------------- Public API ----------------

    def add(self, vecs: np.ndarray, ids: np.ndarray | List[int]) -> List[int]:
        vecs = np.asarray(vecs, dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int64)
        if vecs.ndim != 2 or vecs.shape[1] != self._dim:
            raise ValueError(f"vecs must be [N, {self._dim}] float32")
        if ids.shape[0] != vecs.shape[0]:
            raise ValueError("ids and vecs length mismatch")
        data = [ids.tolist(), vecs.tolist()]
        mr = self._coll.insert(data)
        self._coll.flush()
        return ids.tolist()

    def remove_ids(self, ids: Iterable[int]) -> int:
        ids = list(int(i) for i in ids)
        if len(ids) == 0:
            return 0
        expr = f"id in {ids}"
        self._coll.delete(expr)
        self._coll.flush()
        return len(ids)

    def search(self, q: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if q.shape[1] != self._dim:
            raise ValueError(f"q must have dim={self._dim}")

        res = self._coll.search(
            data=q.tolist(),
            anns_field="emb",
            param=self.cfg.search_param,
            limit=int(k),
            output_fields=["id"]
        )
        # Convert to numpy arrays
        N = len(res)
        D = np.zeros((N, k), dtype=np.float32)
        I = np.full((N, k), -1, dtype=np.int64)
        for i, hits in enumerate(res):
            for j, h in enumerate(hits):
                if j >= k:
                    break
                D[i, j] = float(h.distance)
                I[i, j] = int(h.id)
        # Milvus returns distance; for IP, it's similarity; for L2, it's distance (lower better).
        # Higher-is-better is assumed by downstream—if metric is L2, we may negate distances.
        if self.cfg.metric == "l2":
            D = -D
        return D, I

    # No-op for Milvus: server-side persistence
    def save(self, index_path: str) -> None:
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        with open(index_path + ".milvus.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.cfg), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(index_path: str) -> "MilvusStore":
        cfg_path = index_path + ".milvus.json"
        if not os.path.exists(cfg_path):
            raise FileNotFoundError("Milvus config sidecar not found: " + cfg_path)
        with open(cfg_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        cfg = MilvusConfig(**obj)
        return MilvusStore(cfg)

    # Milvus handles index building; keep API surface compatible
    def maybe_train(self, data=None) -> None:
        return
