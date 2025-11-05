#!/usr/bin/env python3
"""
Build a FAISS index (Flat / IVF / IVF+PQ) from precomputed embeddings.

Typical use:
    python scripts/build_faiss_ivfpq.py \
        --vecs path/to/embeddings.npy \
        --ids path/to/ids.npy \
        --out experiments/results/faiss_index.bin \
        --dim 512 --metric ip --index-spec "IVF4096,PQ64" --nprobe 16

Inputs
- --vecs : .npy file with float32 array shape [N, dim]
- --ids  : optional .npy file with int64 array shape [N]
           (if omitted, IDs are auto-generated starting at 1_000_000)
- --dim  : embedding dimension
- --metric : "ip" or "l2"
- --index-spec : "Flat", "IVF4096,Flat", "IVF4096,PQ64", etc.
- --enforce-normalize : normalize vectors on add/search if metric == ip

Output
- Writes FAISS index binary and sidecar JSON config next to --out.

Notes
- This script bypasses the LabelRegistry; it is purely for building the ANN
  structure (useful for offline experiments or prebuilding large indices).
"""

from __future__ import annotations
import argparse
import os
import sys
import json
import numpy as np

# Allow running from repo root without installing the package
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.memory.faiss_store import FaissStore, FaissConfig


def parse_args():
    ap = argparse.ArgumentParser(description="Build FAISS index from .npy embeddings")
    ap.add_argument("--vecs", required=True, help="Path to embeddings .npy [N, dim] float32")
    ap.add_argument("--ids", default=None, help="Path to ids .npy [N] int64 (optional)")
    ap.add_argument("--out", required=True, help="Output index path (e.g., experiments/results/index.bin)")

    ap.add_argument("--dim", type=int, required=True, help="Embedding dimension")
    ap.add_argument("--metric", choices=["ip", "l2"], default="ip", help="Similarity metric")
    ap.add_argument("--index-spec", default="IVF4096,PQ64", help='Index spec: "Flat", "IVF4096,Flat", "IVF4096,PQ64", ...')
    ap.add_argument("--nprobe", type=int, default=16, help="IVF nprobe at search time")
    ap.add_argument("--train-threshold", type=int, default=200, help="Minimum vectors to trigger training")
    ap.add_argument("--enforce-normalize", action="store_true", help="L2-normalize on add/search when metric=ip")

    ap.add_argument("--chunk", type=int, default=50000, help="Add embeddings in chunks of this size")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    vecs = np.load(args.vecs)
    if vecs.dtype != np.float32:
        vecs = vecs.astype(np.float32, copy=False)
    N, d = vecs.shape
    if d != args.dim:
        raise ValueError(f"Vector dim mismatch: file has {d}, but --dim={args.dim}")

    if args.ids is not None:
        ids = np.load(args.ids).astype(np.int64, copy=False)
        if ids.shape[0] != N:
            raise ValueError("ids length must match number of vectors")
    else:
        start = 1_000_000
        ids = np.arange(start, start + N, dtype=np.int64)

    cfg = FaissConfig(
        dim=args.dim,
        metric=args.metric,
        index_spec=args.index_spec,
        nprobe=args.nprobe,
        enforce_normalize=bool(args.enforce_normalize),
        train_threshold=args.train_threshold,
    )
    store = FaissStore(cfg)

    # Training (for IVF/PQ) on a sample subset
    if not store.is_trained:
        sample = vecs if N <= 100_000 else vecs[np.random.choice(N, 100_000, replace=False)]
        store.maybe_train(sample)

    # Add in chunks to reduce memory spikes
    print(f"[INFO] Building index: N={N}, dim={d}, metric={args.metric}, spec={args.index_spec}")
    step = max(1, int(args.chunk))
    for i in range(0, N, step):
        j = min(N, i + step)
        store.add(vecs[i:j], ids[i:j])
        if (i // step) % 10 == 0:
            print(f"[INFO] Added {j}/{N} vectors...")

    store.save(args.out)
    meta = {
        "num_vectors": int(N),
        "dim": int(d),
        "metric": args.metric,
        "index_spec": args.index_spec,
    }
    with open(args.out + ".build_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved FAISS index → {args.out}")
    print(f"[OK] Config sidecar   → {args.out}.json")
    print(f"[OK] Build metadata   → {args.out}.build_meta.json")


if __name__ == "__main__":
    main()
