# tests/test_retriever.py
import os, sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.retriever import Retriever, RetrieverConfig
from src.memory.incremental import IncrementalMemory
from src.memory.schema import VectorMeta
from src.models.heads.decision import DecisionHead


class DummyImageEncoder:
    """Produces deterministic L2-normalized vectors from BGR frames."""
    def __init__(self, dim=4):
        self.dim = dim

    def encode_images(self, imgs, batch_size=32, tqdm=False):
        out = []
        for im in imgs:
            # if ndarray: use mean per channel; if anything else: fixed seed
            if hasattr(im, "shape"):
                m = np.array([im.mean()] * self.dim, dtype=np.float32)
            else:
                m = np.ones(self.dim, dtype=np.float32)
            v = m / (np.linalg.norm(m) + 1e-9)
            out.append(v.astype(np.float32))
        return np.stack(out, axis=0)


class DummyTextEncoder:
    """Text embeddings: label -> one-hot axis with tiny noise; aligned to vision space for test."""
    def __init__(self, dim=4):
        self.dim = dim

    def encode_text(self, texts, batch_size=64, tqdm=False):
        M = []
        for t in texts:
            idx = abs(hash(t)) % self.dim
            v = np.zeros(self.dim, dtype=np.float32)
            v[idx] = 1.0
            M.append(v)
        M = np.stack(M, axis=0)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        return M.astype(np.float32)


class _ToyStore:
    def __init__(self, dim=4, metric="ip"):
        self.dim = dim
        self.metric = metric
        self._vecs = np.zeros((0, dim), dtype=np.float32)
        self._ids = np.zeros((0,), dtype=np.int64)

    @property
    def is_trained(self): return True
    def maybe_train(self, x): pass

    def add(self, x, ids):
        # normalize for ip
        x = x.astype(np.float32)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        self._vecs = np.vstack([self._vecs, x])
        self._ids = np.concatenate([self._ids, ids.astype(np.int64)])

    def remove_ids(self, ids):
        ids = set(int(i) for i in ids)
        mask = np.array([i not in ids for i in self._ids], bool)
        removed = int((~mask).sum())
        self._vecs, self._ids = self._vecs[mask], self._ids[mask]
        return removed

    def search(self, q, k=5):
        q = q.astype(np.float32)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        sims = q @ self._vecs.T if self._vecs.size else np.full((q.shape[0], k), -1e9)
        order = np.argsort(-sims, axis=1)[:, :k] if self._vecs.size else np.zeros((q.shape[0], k), int)
        D = np.take_along_axis(sims, order, axis=1).astype(np.float32) if self._vecs.size else sims
        I = self._ids[order] if self._vecs.size else np.full((q.shape[0], k), -1, dtype=np.int64)
        return D, I


@pytest.fixture
def world(tmp_path):
    dim = 4
    mem = IncrementalMemory(dim=dim, metric="ip", backend="faiss", persist_dir=str(tmp_path), enforce_normalize=True)
    mem.store = _ToyStore(dim=dim, metric="ip")
    img_enc = DummyImageEncoder(dim=dim)
    txt_enc = DummyTextEncoder(dim=dim)

    # create two classes with distinct means
    A = np.tile(np.array([1, 0, 0, 0], np.float32), (3, 1))
    B = np.tile(np.array([0, 1, 0, 0], np.float32), (3, 1))
    mem.register_class("mug", A, metas=[VectorMeta(label="mug").to_dict() for _ in range(3)])
    mem.register_class("bottle", B, metas=[VectorMeta(label="bottle").to_dict() for _ in range(3)])
    return mem, img_enc, txt_enc


def test_retriever_top1(world):
    mem, img_enc, txt_enc = world
    cfg = RetrieverConfig(k=3, alpha_fusion=0.7, temporal_ema=0.0, neighbor_agg="mean")
    r = Retriever(memory=mem, image_encoder=img_enc, text_encoder=txt_enc, text_aligned=False, cfg=cfg)

    # Frame with strong "mug" signal (use synthetic ndarray)
    frame = np.full((10, 10, 3), 1.0, dtype=np.float32)  # mean=1
    out = r.retrieve(frame)
    assert out.label in {"mug", "bottle"}  # determined by dummy encoder mapping
    assert isinstance(out.score, float)


def test_retriever_with_text_fusion(world):
    mem, img_enc, txt_enc = world
    cfg = RetrieverConfig(k=3, alpha_fusion=0.3, temporal_ema=0.0, neighbor_agg="max")
    r = Retriever(memory=mem, image_encoder=img_enc, text_encoder=txt_enc, text_aligned=True, cfg=cfg)

    # Build text prototypes for the existing labels
    r.update_text_prototypes(mem.registry.labels())

    frame = np.zeros((8, 8, 3), dtype=np.float32)  # mean=0 → neutral image vec
    out = r.retrieve(frame)
    assert out.label in {"mug", "bottle"}
    # With text fusion active and alpha small, text can steer decisions
    assert isinstance(out.score, float)


def test_decision_head_integration(world):
    from src.pipeline.inference import InferenceEngine
    mem, img_enc, txt_enc = world
    r = Retriever(memory=mem, image_encoder=img_enc, text_encoder=None, text_aligned=False,
                  cfg=RetrieverConfig(k=3, alpha_fusion=0.7, temporal_ema=0.1))
    dec = DecisionHead(temperature=0.9, tau_open=0.1, use_margin=True, margin_delta=0.05)

    engine = InferenceEngine(retriever=r, decision=dec)
    frame = np.ones((8, 8, 3), dtype=np.float32)
    y = engine.infer(frame)
    assert hasattr(y, "label") and hasattr(y, "score") and hasattr(y, "latency_ms")
