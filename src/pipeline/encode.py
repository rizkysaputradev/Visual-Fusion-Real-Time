# src/pipeline/encode.py
"""
Vision-Fusion-RT — Encoding Utilities
-------------------------------------

Thin wrappers around the vision & text encoders to keep the rest of the codebase
framework-agnostic and consistent.

Functions
- encode_images_bgr(img_enc, images_bgr, batch_size=32, tqdm=False) -> np.ndarray [N, d]
- encode_texts(txt_enc, texts, batch_size=64, tqdm=False) -> np.ndarray [M, d]
- ensure_l2(x) -> np.ndarray [*, d]   (normalize, defensive)

Notes
- Accepts OpenCV BGR ndarrays or PIL; the underlying encoders already handle both.
- Returns float32, L2-normalized embeddings (encoders already normalize; we re-assert).
"""

from __future__ import annotations
from typing import Iterable, List, Sequence
import numpy as np

from src.core.utils import l2_normalize


def encode_images_bgr(image_encoder, images_bgr: Sequence, batch_size: int = 32, tqdm: bool = False) -> np.ndarray:
    """
    Encode a list of images (BGR or PIL). Returns float32 [N, d].
    """
    if len(images_bgr) == 0:
        # Infer dim by encoding a single dummy (encoders generally need real input—so return empty)
        return np.zeros((0, getattr(image_encoder, "dim", 0)), dtype=np.float32)
    vecs = image_encoder.encode_images(list(images_bgr), batch_size=batch_size, tqdm=tqdm)
    return ensure_l2(vecs)


def encode_texts(text_encoder, texts: Sequence[str], batch_size: int = 64, tqdm: bool = False) -> np.ndarray:
    """
    Encode a list of texts. Returns float32 [M, d].
    """
    if text_encoder is None or len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    vecs = text_encoder.encode_text(list(texts), batch_size=batch_size, tqdm=tqdm)
    return ensure_l2(vecs)


def ensure_l2(x: np.ndarray) -> np.ndarray:
    """
    Enforce float32 + L2 normalization defensively.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    return l2_normalize(x, axis=1)
