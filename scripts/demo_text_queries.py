#!/usr/bin/env python3
"""
Demo: Text prototypes & query similarity against known labels.

Use cases
- Inspect how your current label set aligns with a chosen text encoder (CLIP or HF).
- Try different prompt templates to see which give better separation.

Examples
---------
1) Using a registry to get labels + CLIP text encoder:
    python scripts/demo_text_queries.py \
        --labels-json data/registries/mem_labels.json.gz \
        --text-encoder clip_text \
        --clip-model "openai/clip-vit-base-patch32" \
        --device cuda \
        --template "a photo of a {label}" \
        --queries "red mug, black backpack, blue water bottle" \
        --topk 5 \
        --out-csv experiments/results/text_query_demo.csv

2) Using HF MiniLM text encoder (semantic, *not CLIP-aligned*):
    python scripts/demo_text_queries.py \
        --labels-json data/registries/mem_labels.json.gz \
        --text-encoder hf_minilm \
        --device cpu \
        --template "{label}" \
        --queries "mug, backpack, bottle"

Outputs
- Prints a neat table of cosine similarities query↔label_prototype.
- Optionally writes a CSV with the full matrix.

Notes
- If you use a non-aligned HF encoder, this is just a *semantic* similarity—not
  directly comparable to image space. Use it to craft better prompts/labels.
"""

from __future__ import annotations
import argparse
import os
import sys
import json
import gzip
import csv
import numpy as np

# repo sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.text.registry import build_text_encoder
from src.memory.schema import LabelRegistry


def _load_labels_from_registry(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # Use LabelRegistry loader to be consistent; but we only need labels
    reg = LabelRegistry.load(path)
    return reg.labels()


def _l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def parse_args():
    ap = argparse.ArgumentParser(description="Demo text prototypes vs. queries")
    ap.add_argument("--labels-json", required=True, help="Path to saved registry JSON(.gz)")
    ap.add_argument("--text-encoder", default="clip_text", help='One of: "clip_text", "hf_text", "hf_minilm", "hf_e5_small"')
    ap.add_argument("--device", default="cpu", help='Device for text encoder: "cuda", "mps", or "cpu"')

    # CLIP text specific
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32", help="HF CLIP model id (text tower)")

    # Prompting / queries
    ap.add_argument("--template", default="a photo of a {label}", help="Prompt template for label text")
    ap.add_argument("--queries", required=True, help='Comma-separated queries, e.g. "red mug, black backpack"')

    ap.add_argument("--topk", type=int, default=5, help="Top-k labels to show per query")
    ap.add_argument("--out-csv", default=None, help="Optional CSV path to dump full similarity matrix")
    return ap.parse_args()


def main():
    args = parse_args()
    labels = _load_labels_from_registry(args.labels_json)
    if len(labels) == 0:
        print("[WARN] No labels found in registry. Add classes first.")
        return

    # Build text encoder
    extra = {}
    if args.text_encoder == "clip_text":
        extra["model_name"] = args.clip_model
    txt = build_text_encoder(args.text_encoder, device=args.device, **extra)

    # Build label prompts and encode
    label_prompts = [args.template.format(label=lbl) for lbl in labels]
    V = txt.encode_text(label_prompts, batch_size=max(8, len(labels)))
    V = _l2norm(V)  # [L, d]

    # Queries
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    Q = txt.encode_text(queries, batch_size=len(queries))
    Q = _l2norm(Q)  # [Q, d]

    # Cosine similarity = dot product (both normalized)
    S = Q @ V.T  # [Q, L]

    # Print per query top-k
    print(f"\n[INFO] Labels: {len(labels)} | Queries: {len(queries)} | Encoder: {args.text_encoder}")
    for i, q in enumerate(queries):
        sims = S[i]
        order = np.argsort(-sims)[:max(1, args.topk)]
        print(f"\nQuery: {q}")
        for rank, j in enumerate(order, start=1):
            print(f"  {rank:>2}. {labels[j]:<24}  sim={float(sims[j]):.4f}")

    # Optional CSV dump of full matrix
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query"] + labels)
            for i, q in enumerate(queries):
                writer.writerow([q] + [f"{float(S[i, j]):.6f}" for j in range(len(labels))])
        print(f"\n[OK] Wrote full similarity matrix → {args.out_csv}")


if __name__ == "__main__":
    main()
