# tests/test_register_flow.py
import os, sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.register import register_from_frames
from src.memory.incremental import IncrementalMemory
from src.memory.schema import VectorMeta
from src.models.preproc.augment import Augmenter


class DummyImageEncoder:
    def __init__(self, dim=8): self.dim = dim
    def encode_images(self, imgs, batch_size=32, tqdm=False):
        out = []
        for i, im in enumerate(imgs):
            # deterministic embedding per image content/shape
            seed = int(np.mean(im) * 1000) if hasattr(im, "shape") else i
            rng = np.random.default_rng(seed)
            v = rng.normal(0, 1, size=(self.dim,)).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-9)
            out.append(v)
        return np.stack(out, axis=0)


class _ToyStore:
    def __init__(self, dim=8, metric="ip"):
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
    def save(self, path):  # minimal parity
        np.savez_compressed(path + ".npz", vecs=self._vecs, ids=self._ids)
        return path + ".npz"

@pytest.fixture
def mem(tmp_path):
    dim = 8
    m = IncrementalMemory(dim=dim, metric="ip", backend="faiss", persist_dir=str(tmp_path), enforce_normalize=True)
    m.store = _ToyStore(dim=dim, metric="ip")
    return m

def test_register_from_frames_basic(mem):
    enc = DummyImageEncoder(dim=8)
    # Create 4 synthetic frames with distinct means
    frames = [np.full((16,16,3), fill_value=v, dtype=np.float32) for v in (0.1, 0.2, 0.3, 0.4)]
    added = register_from_frames(memory=mem, img_enc=enc, frames_bgr=frames, label="stapler", augmenter=None, batch_size=4)
    assert added >= 4
    assert "stapler" in mem.registry.labels()
    assert mem.get_centroid("stapler") is not None

def test_register_with_augmentation(mem):
    enc = DummyImageEncoder(dim=8)
    frames = [np.full((8,8,3), 0.5, np.float32) for _ in range(3)]
    aug = Augmenter("low")
    n = register_from_frames(memory=mem, img_enc=enc, frames_bgr=frames, label="mouse", augmenter=aug, batch_size=3)
    # with aug we add at least original+augmented samples
    assert n >= len(frames)
    assert "mouse" in mem.registry.labels()
