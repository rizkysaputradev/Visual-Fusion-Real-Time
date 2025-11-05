# src/memory/store_router.py
"""
Vision-Fusion-RT — Store Router
-------------------------------

Picks the memory backend (FAISS or Milvus) based on `AppConf.memory` and
returns a ready-to-use store instance exposing a unified API:
    - dim, ntotal
    - add(vecs, ids) -> List[int]
    - remove_ids(ids) -> int
    - search(q, k) -> (D, I)
    - save(path) / load(path)
    - maybe_train(data)

Usage
    from src.memory.store_router import build_store
    store = build_store(dim=512, backend="faiss", metric="ip", index_spec="IVF4096,PQ64")
"""

from __future__ import annotations
from typing import Optional, Dict, Any

from .faiss_store import FaissStore, FaissConfig
from .milvus_store import MilvusStore, MilvusConfig


def build_store(dim: int,
                backend: str = "faiss",
                metric: str = "ip",
                index_spec: str = "Flat",
                **kwargs):
    backend = backend.lower()
    if backend == "faiss":
        cfg = FaissConfig(dim=dim, metric=metric, index_spec=index_spec,
                          nprobe=int(kwargs.get("nprobe", 16)),
                          enforce_normalize=bool(kwargs.get("enforce_normalize", False)),
                          train_threshold=int(kwargs.get("train_threshold", 200)))
        return FaissStore(cfg)
    elif backend == "milvus":
        cfg = MilvusConfig(
            dim=dim,
            metric=metric,
            host=str(kwargs.get("host", "127.0.0.1")),
            port=str(kwargs.get("port", "19530")),
            collection_name=str(kwargs.get("collection_name", "vision_fusion_rt")),
            index_type=str(kwargs.get("index_type", "IVF_FLAT")),
            index_param=kwargs.get("index_param"),
            search_param=kwargs.get("search_param"),
        )
        return MilvusStore(cfg)
    else:
        raise KeyError(f"Unknown backend '{backend}'. Use 'faiss' or 'milvus'.")
