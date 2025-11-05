# tests/test_memory_store.py
import os, sys, time
from pathlib import Path
import numpy as np
import pytest

# Make src importable when running from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.memory.schema import LabelRegistry, VectorMeta
from src.memory.incremental import IncrementalMemory


# -----------------------------
# A tiny, dependency-free ANN store used for tests
# -----------------------------
class _ToyStore:
    """
    Minimal vector store with FAISS-like API: add(), remove_ids(), search().
    Uses brute-force numpy search. Metric: "ip" or "l2".
    """
    def __init__(self, dim: int, metric: str = "ip", enforce_normalize: bool = True):
        self.dim = int(dim)
        self.metric = metric
        self.enforce_normalize = bool(enforce_normalize)
        self._vecs = np.zeros((0, dim), dtype=np.float32)
        self._ids = np.zeros((0,), dtype=np.int64)
        self._trained = True

    @property
    def is_trained(self):  # parity with FaissStore
        return True

    def maybe_train(self, x):  # no-op
        pass

    def add(self, x: np.ndarray, ids: np.ndarray) -> None:
        x = np.asanyarray(x, dtype=np.float32)
        if self.enforce_normalize and self.metric == "ip":
            n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
            x = x / n
        self._vecs = np.vstack([self._vecs, x])
        self._ids = np.concatenate([self._ids, ids.astype(np.int64)])

    def remove_ids(self, ids: np.ndarray) -> int:
        ids = set(int(i) for i in ids)
        mask = np.array([i not in ids for i in self._ids], dtype=bool)
        removed = int((~mask).sum())
        self._vecs = self._vecs[mask]
        self._ids = self._ids[mask]
        return removed

    def search(self, q: np.ndarray, k: int = 5):
        q = np.asanyarray(q, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if self.enforce_normalize and self.metric == "ip":
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        if self._vecs.shape[0] == 0:
            D = np.full((q.shape[0], k), -1e9 if self.metric == "ip" else 1e9, dtype=np.float32)
            I = np.full((q.shape[0], k), -1, dtype=np.int64)
            return D, I

        if self.metric == "ip":
            sims = q @ self._vecs.T  # [N, M]
            order = np.argsort(-sims, axis=1)[:, :k]
            D = np.take_along_axis(sims, order, axis=1).astype(np.float32)
        else:  # l2
            # (q - x)^2 = q^2 + x^2 - 2 q·x
            q2 = (q ** 2).sum(axis=1, keepdims=True)
            x2 = (self._vecs ** 2).sum(axis=1, keepdims=True).T
            dist = q2 + x2 - 2.0 * (q @ self._vecs.T)
            order = np.argsort(dist, axis=1)[:, :k]
            D = np.take_along_axis(dist, order, axis=1).astype(np.float32)

        I = self._ids[order]
        return D, I

    # parity with FaissStore API used by IncrementalMemory persistence
    def save(self, path: str) -> str:
        np.savez_compressed(path + ".npz", vecs=self._vecs, ids=self._ids, allow_pickle=False)
        # we also return a JSON sidecar in FaissStore; here just return base path
        return path + ".npz"

    @classmethod
    def load(cls, path: str, dim: int, metric: str = "ip", enforce_normalize: bool = True):
        obj = cls(dim=dim, metric=metric, enforce_normalize=enforce_normalize)
        data = np.load(path + ".npz")
        obj._vecs = data["vecs"].astype(np.float32)
        obj._ids = data["ids"].astype(np.int64)
        return obj


@pytest.fixture
def toy_memory(tmp_path):
    # build IncrementalMemory but inject our toy store
    dim = 4
    mem = IncrementalMemory(
        dim=dim,
        metric="ip",
        backend="faiss",          # value not used after we inject the store
        persist_dir=str(tmp_path),
        enforce_normalize=True,
    )
    mem.store = _ToyStore(dim=dim, metric="ip", enforce_normalize=True)
    return mem


def test_label_registry_basic(tmp_path):
    reg = LabelRegistry()
    # Simulate two vectors with labels
    reg.add(vid=1, meta=VectorMeta(label="mug", ts=time.time(), source="test"))
    reg.add(vid=2, meta=VectorMeta(label="bottle", ts=time.time(), source="test"))
    assert reg.meta_of(1).label == "mug"
    assert set(reg.labels()) == {"mug", "bottle"}

    # Save & load
    out = reg.save(tmp_path / "mem_labels.json.gz")
    reg2 = LabelRegistry.load(out)
    assert reg2.meta_of(1).label == "mug"
    assert set(reg2.labels()) == {"mug", "bottle"}


def test_incremental_memory_register_and_search(toy_memory):
    mem = toy_memory
    # Make two tiny clusters in 4D
    rng = np.random.default_rng(0)
    mug = rng.normal(0.0, 0.05, size=(5, 4)).astype(np.float32) + np.array([1, 0, 0, 0], dtype=np.float32)
    bottle = rng.normal(0.0, 0.05, size=(5, 4)).astype(np.float32) + np.array([0, 1, 0, 0], dtype=np.float32)

    ids1 = mem.register_class("mug", mug, metas=[VectorMeta(label="mug").to_dict() for _ in range(len(mug))])
    ids2 = mem.register_class("bottle", bottle, metas=[VectorMeta(label="bottle").to_dict() for _ in range(len(bottle))])
    assert len(ids1) == 5 and len(ids2) == 5
    assert set(mem.registry.labels()) == {"mug", "bottle"}

    # Query near mug direction
    q = np.array([1, 0, 0, 0], dtype=np.float32)
    D, I = mem.store.search(q[None, :], k=3)
    assert I.shape == (1, 3)
    # Top neighbor should be a mug vector
    top_vid = int(I[0, 0])
    assert mem.registry.meta_of(top_vid).label == "mug"

    # Test undo: remove last 2 for mug
    removed = mem.undo_last_for("mug", n=2)
    assert removed == 2
    assert len([vid for vid, m in mem.registry._id2meta.items() if m.label == "mug"]) == 3


def test_persistence_roundtrip(tmp_path):
    # Combine registry + store roundtrip
    mem = IncrementalMemory(dim=3, metric="ip", backend="faiss", persist_dir=str(tmp_path), enforce_normalize=True)
    mem.store = _ToyStore(dim=3, metric="ip", enforce_normalize=True)

    # Add a few
    X = np.eye(3, dtype=np.float32)
    ids = mem.register_class("axis", X, metas=[VectorMeta(label="axis").to_dict() for _ in range(3)])
    assert len(ids) == 3

    # Save
    idx_path = mem.save_store("toy_index")
    reg_path = mem.save_registry("toy_labels.json.gz")
    assert Path(idx_path).exists() or Path(idx_path + ".npz").exists()
    assert Path(reg_path).exists()

    # Load back into a new memory
    mem2 = IncrementalMemory(dim=3, metric="ip", backend="faiss", persist_dir=str(tmp_path), enforce_normalize=True)
    mem2.store = _ToyStore.load(idx_path.replace(".npz", ""), dim=3, metric="ip", enforce_normalize=True)
    mem2.registry = LabelRegistry.load(reg_path)

    D, I = mem2.store.search(np.array([1, 0, 0], dtype=np.float32)[None, :], k=1)
    assert int(I[0, 0]) in mem2.registry._id2meta
