# tests/test_rt_loop.py
import os, sys, time
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.bench_rt import run_realtime_benchmark
from src.retrieval.retriever import Retriever, RetrieverConfig
from src.models.heads.decision import DecisionHead
from src.pipeline.inference import InferenceEngine
from src.memory.incremental import IncrementalMemory
from src.memory.schema import VectorMeta


class DummyImageEncoder:
    def __init__(self, dim=4): self.dim = dim
    def encode_images(self, imgs, batch_size=32, tqdm=False):
        out = []
        for im in imgs:
            m = float(np.mean(im)) if hasattr(im, "shape") else 0.0
            v = np.array([m, 1-m, 0, 0], dtype=np.float32)
            v /= (np.linalg.norm(v) + 1e-9)
            out.append(v)
        return np.stack(out).astype(np.float32)


class _ToyStore:
    def __init__(self, dim=4, metric="ip"):
        self.dim = dim; self.metric = metric
        self._vecs = np.zeros((0, dim), np.float32); self._ids = np.empty((0,), np.int64)
    @property
    def is_trained(self): return True
    def maybe_train(self, x): pass
    def add(self, x, ids):
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        self._vecs = np.vstack([self._vecs, x]); self._ids = np.concatenate([self._ids, ids.astype(np.int64)])
    def remove_ids(self, ids):
        ids = set(int(i) for i in ids)
        mask = np.array([i not in ids for i in self._ids], bool)
        removed = int((~mask).sum())
        self._vecs, self._ids = self._vecs[mask], self._ids[mask]
        return removed
    def search(self, q, k=5):
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        sims = q @ self._vecs.T if self._vecs.size else np.full((q.shape[0], k), -1e9)
        order = np.argsort(-sims, axis=1)[:, :k] if self._vecs.size else np.zeros((q.shape[0], k), int)
        D = np.take_along_axis(sims, order, axis=1).astype(np.float32) if self._vecs.size else sims
        I = self._ids[order] if self._vecs.size else np.full((q.shape[0], k), -1, dtype=np.int64)
        return D, I


@pytest.fixture
def engine(tmp_path):
    dim = 4
    mem = IncrementalMemory(dim=dim, metric="ip", backend="faiss", persist_dir=str(tmp_path), enforce_normalize=True)
    mem.store = _ToyStore(dim=dim, metric="ip")

    # two classes
    A = np.tile(np.array([1, 0, 0, 0], np.float32), (5, 1))
    B = np.tile(np.array([0, 1, 0, 0], np.float32), (5, 1))
    mem.register_class("bright", A, metas=[VectorMeta(label="bright").to_dict() for _ in range(5)])
    mem.register_class("dark", B, metas=[VectorMeta(label="dark").to_dict() for _ in range(5)])

    img_enc = DummyImageEncoder(dim=dim)
    retr = Retriever(memory=mem, image_encoder=img_enc, text_encoder=None, text_aligned=False,
                     cfg=RetrieverConfig(k=5, alpha_fusion=0.7, temporal_ema=0.05))
    dec = DecisionHead(temperature=0.9, tau_open=0.05, use_margin=False)
    return InferenceEngine(retriever=retr, decision=dec)


def test_rt_loop_benchmark_monkeypatched(engine, monkeypatch):
    """
    run_realtime_benchmark uses an internal _open_source() -> (read_latest, close).
    We monkeypatch it to stream synthetic frames quickly.
    """
    from src.eval import bench_rt

    frames = [np.full((32,32,3), v, np.float32) for v in (0.0, 0.2, 0.5, 0.8, 1.0)]
    idx = {"i": 0}

    def fake_open_source(source, size):
        def read_latest():
            i = idx["i"]
            idx["i"] = (i + 1) % len(frames)
            return frames[i]
        def close():
            pass
        return read_latest, close

    monkeypatch.setattr(bench_rt, "_open_source", fake_open_source)

    stats = bench_rt.run_realtime_benchmark(
        engine=engine,
        source="webcam://0",   # value irrelevant due to monkeypatch
        seconds=1,             # short run
        target_size=(64, 48),
        warmup_s=0.1,
        display=False,
        show_overlay=False
    )
    assert stats["n_frames"] > 0
    assert stats["fps_mean"] >= 0.0
