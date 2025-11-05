# src/io/image_io.py
"""
Vision-Fusion-RT — Image I/O Utilities
--------------------------------------

Responsibilities
- Robust disk read/write wrappers for images (OpenCV BGR).
- PIL ↔ OpenCV conversion helpers (RGB↔BGR).
- Folder utilities for dataset scanning.
- EXIF orientation correction for PIL inputs (optional).

Notes
- All public read functions return **BGR** ndarray (OpenCV convention).
- `imwrite` guarantees parent directory creation.
"""

from __future__ import annotations
import os
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


# -------------------------
# Core read/write
# -------------------------

def imread(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """
    Read an image from disk as BGR ndarray (uint8). Raises on failure.
    """
    img = cv2.imread(path, flags)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def imwrite(path: str, img: np.ndarray) -> None:
    """
    Write BGR ndarray to disk (creates parent folders). Raises on failure.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok = cv2.imwrite(path, img)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


# -------------------------
# Conversions
# -------------------------

def to_pil(bgr: np.ndarray, correct_exif: bool = False) -> Image.Image:
    """
    Convert BGR ndarray → PIL RGB Image. Optionally apply EXIF orientation.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    if correct_exif:
        im = ImageOps.exif_transpose(im)
    return im


def to_bgr(pil: Image.Image, correct_exif: bool = False) -> np.ndarray:
    """
    Convert PIL RGB (or L) Image → BGR ndarray. Optionally apply EXIF orientation first.
    """
    if correct_exif:
        pil = ImageOps.exif_transpose(pil)
    rgb = pil.convert("RGB")
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    return bgr


# -------------------------
# Bulk & Folder helpers
# -------------------------

def list_images(folder: str, exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".gif")) -> List[str]:
    """
    List image file paths in a folder (non-recursive), sorted lexicographically.
    """
    entries = []
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and name.lower().endswith(tuple(exts)):
            entries.append(p)
    return entries


def load_folder(folder: str, exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp")) -> List[np.ndarray]:
    """
    Load all images in a folder (non-recursive) as BGR arrays.
    """
    return [imread(p) for p in list_images(folder, exts=exts)]


def save_grid(path: str, images_bgr: Sequence[np.ndarray], cols: int = 4, pad: int = 4) -> None:
    """
    Save a grid image for quick dataset inspection.

    Parameters
    ----------
    path : str
        Output path (PNG/JPG).
    images_bgr : Sequence[np.ndarray]
        List of BGR images (can have varying sizes).
    cols : int
        Number of columns.
    pad : int
        Padding (px) between tiles.
    """
    if len(images_bgr) == 0:
        raise ValueError("No images provided to save_grid().")

    # Normalize sizes: pick tile size = median (width, height)
    Hs = [im.shape[0] for im in images_bgr]
    Ws = [im.shape[1] for im in images_bgr]
    Hm = sorted(Hs)[len(Hs) // 2]
    Wm = sorted(Ws)[len(Ws) // 2]

    tiles = [cv2.resize(im, (Wm, Hm), interpolation=cv2.INTER_AREA) for im in images_bgr]

    rows = (len(tiles) + cols - 1) // cols
    grid_h = rows * Hm + (rows + 1) * pad
    grid_w = cols * Wm + (cols + 1) * pad

    canvas = np.full((grid_h, grid_w, 3), 245, dtype=np.uint8)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(tiles):
                break
            y = r * Hm + (r + 1) * pad
            x = c * Wm + (c + 1) * pad
            canvas[y:y+Hm, x:x+Wm] = tiles[idx]
            idx += 1

    imwrite(path, canvas)
