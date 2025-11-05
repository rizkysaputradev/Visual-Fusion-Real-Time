#!/usr/bin/env python3
"""
Bulk-register all images in a folder as a single class label.

Typical use:
    python scripts/import_folder_as_class.py \
        --folder data/samples/mug \
        --label "mug" \
        --backbone clip_vit_b32 \
        --device cuda \
        --backend faiss \
        --metric ip \
        --index-spec "IVF4096,PQ64" \
        --persist-dir data/registries \
        --limit 500 \
        --augment mid

What it does
- Loads/initializes an IncrementalMemory (FAISS or Milvus backend).
- Builds the image encoder (CLIP/ViT via backbones registry).
- Reads all images in the folder (non-recursive), encodes them,
  applies light augmentations if requested, and registers them
  under the given label.
- Saves the vector index and label registry to --persist-dir.

Outputs
- <persist-dir>/mem_index.bin (+ .json sidecar)
- <persist-dir>/mem_labels.json.gz

You can invoke this repeatedly with different labels/folders to grow the memory.
"""

from __future__ import annotations
import argparse
import os
import sys
import time

# repo import path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.backbones.registry import build_backbone, list_backbones
from src.models.preproc.augment import Augmenter
from src.memory.incremental import IncrementalMemory
from src.pipeline.register import register_from_folder
from src.io.image_io import list_images


def parse_args():
    ap = argparse.ArgumentParser(description="Import a folder of images as one class label")
    ap.add_argument("--folder", required=True, help="Path to folder containing class images (non-recursive)")
    ap.add_argument("--label", required=True, help="Class label name")

    # encoders / device
    ap.add_argument("--backbone", default="clip_vit_b32", help=f"Backbone key (choices: {', '.join(list_backbones())})")
    ap.add_argument("--device", default="cpu", help='Device for encoder: "cuda", "mps", or "cpu"')
    ap.add_argument("--fp16", action="store_true", help="Use FP16 autocast where available")

    # memory backend
    ap.add_argument("--backend", choices=["faiss", "milvus"], default="faiss")
    ap.add_argument("--metric", choices=["ip", "l2"], default="ip")
    ap.add_argument("--index-spec", default="IVF4096,PQ64")
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--persist-dir", default="data/registries", help="Where to save index & registry")

    # data controls
    ap.add_argument("--limit", type=int, default=None, help="Limit number of images read from the folder")
    ap.add_argument("--augment", choices=["none", "low", "mid", "high"], default="none",
                    help="Apply light augmentations to boost robustness")
    ap.add_argument("--batch-size", type=int, default=64, help="Batch size for encoding")
    return ap.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.folder):
        raise FileNotFoundError(f"Folder not found: {args.folder}")

    # Build encoder
    enc = build_backbone(args.backbone, device=args.device, fp16=args.fp16)
    dim = int(enc.dim)

    # Create memory manager
    mem = IncrementalMemory(
        dim=dim,
        metric=args.metric,
        backend=args.backend,
        index_spec=args.index_spec,
        persist_dir=args.persist_dir,
        nprobe=args.nprobe,
        enforce_normalize=(args.metric == "ip"),
    )

    # Augmentation (optional)
    augmenter = None if args.augment == "none" else Augmenter(str(args.augment))

    # Register
    n_imgs = len(list_images(args.folder))
    if args.limit is not None:
        n_imgs = min(n_imgs, int(args.limit))
    print(f"[INFO] Registering '{args.label}' from {args.folder} (n={n_imgs}) with backbone={args.backbone} (dim={dim})")
    added = register_from_folder(
        memory=mem,
        img_enc=enc,
        folder=args.folder,
        label=args.label,
        limit=args.limit,
        augmenter=augmenter,
        batch_size=args.batch_size,
    )
    print(f"[OK] Added {added} vectors for label='{args.label}'")

    # Persist
    out_idx = mem.save_store("mem_index.bin")
    out_reg = mem.save_registry("mem_labels.json.gz")
    print(f"[OK] Saved index   → {out_idx}")
    print(f"[OK] Saved labels  → {out_reg}")


if __name__ == "__main__":
    main()
