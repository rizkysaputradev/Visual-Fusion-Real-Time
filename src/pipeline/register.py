# src/pipeline/register.py
"""
Vision-Fusion-RT — Few-Shot Registration Flow
---------------------------------------------

High-level routines to register new classes online:

- register_from_frames(memory, img_enc, frames_bgr, label, augmenter=None)
- register_from_folder(memory, img_enc, folder, label, limit=None, augmenter=None)
- register_text_prototypes(retriever, labels)  # convenience

Behavior
- Converts incoming frames/images into embeddings via `encode.py`.
- Optional light augmentations (see models/preproc/augment.py) to boost robustness.
- Writes embeddings into the vector store through `IncrementalMemory`.
- Computes and stores a per-class centroid (mean embedding) for downstream fusion.

Thread-safety
- `IncrementalMemory` is already synchronized; these helpers are stateless wrappers.

"""

from __future__ import annotations
from typing import Iterable, List, Optional
import os
import time
import numpy as np
from PIL import Image

from src.pipeline.encode import encode_images_bgr, ensure_l2
from src.io.image_io import list_images, imread, to_pil
from src.memory.incremental import IncrementalMemory
from src.memory.schema import VectorMeta


def _to_pils(frames_or_paths: List) -> List[Image.Image]:
    out: List[Image.Image] = []
    for item in frames_or_paths:
        if isinstance(item, str):
            bgr = imread(item)
            out.append(to_pil(bgr))
        elif isinstance(item, np.ndarray):
            out.append(to_pil(item))
        elif isinstance(item, Image.Image):
            out.append(item.convert("RGB"))
        else:
            raise TypeError(f"Unsupported input type: {type(item)}")
    return out


def register_from_frames(
    memory: IncrementalMemory,
    img_enc,
    frames_bgr: List[np.ndarray] | List[Image.Image],
    label: str,
    augmenter=None,
    batch_size: int = 32,
) -> int:
    """
    Register a new/updated class from a handful of frames.
    Returns number of vectors added.
    """
    pils = _to_pils(frames_bgr)

    # Optional augmentations (PIL-in, PIL-out)
    if augmenter is not None:
        pils = list(pils) + augmenter.apply(pils, n_per_image=2)  # small boost

    # Encode to embeddings (image encoder handles PIL/ndarray)
    # We pass PILs so encoder-specific processors (CLIP) behave best.
    vecs = img_enc.encode_images(pils, batch_size=batch_size)
    vecs = ensure_l2(vecs)

    # Persist via memory manager
    now = time.time()
    metas = [VectorMeta(label=label, ts=now, source="rt").to_dict() for _ in range(vecs.shape[0])]
    ids = memory.register_class(label, vecs, metas=metas)

    # Compute centroid and store (exact mean of current registration batch;
    # if you want the true centroid across *all* label vectors, you can extend
    # the FaissStore with `.reconstruct(ids)` and recompute precisely.)
    centroid = vecs.mean(axis=0, dtype=np.float32)
    memory.set_centroid(label, centroid)

    return len(ids)


def register_from_folder(
    memory: IncrementalMemory,
    img_enc,
    folder: str,
    label: str,
    limit: Optional[int] = None,
    augmenter=None,
    batch_size: int = 64,
) -> int:
    """
    Bulk register samples from a folder of images (non-recursive).
    """
    paths = list_images(folder)
    if limit is not None:
        paths = paths[: int(limit)]

    pils = _to_pils(paths)

    if augmenter is not None:
        pils = list(pils) + augmenter.apply(pils, n_per_image=1)

    vecs = img_enc.encode_images(pils, batch_size=batch_size, tqdm=True)
    vecs = ensure_l2(vecs)

    now = time.time()
    metas = [VectorMeta(label=label, ts=now, source="file", attrs={"path": p}).to_dict() for p in paths] + \
            [VectorMeta(label=label, ts=now, source="aug").to_dict() for _ in range(len(pils) - len(paths))]

    ids = memory.register_class(label, vecs, metas=metas)
    memory.set_centroid(label, vecs.mean(axis=0, dtype=np.float32))
    return len(ids)


def register_text_prototypes(retriever, labels: List[str]) -> None:
    """
    Convenience: keep text prototypes synchronized with known labels.
    """
    retriever.update_text_prototypes(labels)
