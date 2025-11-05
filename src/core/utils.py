### ========================================================================================================================================
## Module       : src/core/utils.py
## Author       : Rizky Johan Saputra (Independent Project)
## Date         : 4th November 2025 (Seoul, South Korea)
## Project      : Vision Fusion Real Time System (Copyright 2025)
## Topics       : Computer Vision, Real-Time Systems, Interactive AI System, NLP, Machine Learning and Memory Augmentation
## Purpose      : 
## Role         : Typed Core Structures
### ========================================================================================================================================

## ======================================================================================================
## SPECIFICATIONS
## ======================================================================================================
"""
Vision-Fusion-RT — Core Utilities
---------------------------------

- Device selection helper (cuda/mps/cpu) without importing heavy libs elsewhere.
- Seeding RNGs (Python, NumPy, PyTorch if available).
- L2 normalization utility.
- Small helpers: chunked iteration, clamp, safe import check.
"""

## ======================================================================================================
## SETUP (ADJUSTABLE) (ADJUST IF NECESSARY)
## ======================================================================================================
from __future__ import annotations
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar
import random
import os
import numpy as np

## ======================================================================================================
## IMPLEMENTATIONS
## ======================================================================================================
#
T = TypeVar("T")

#
def get_device(pref: str = "cuda"):
    """
    Return a torch.device-like object *string* depending on availability.
    We do not import torch here to keep this module lightweight; callers who need
    real torch.device should convert this string using torch APIs.

    Returns one of: "cuda" | "mps" | "cpu" (best effort).
    """
    try:
        import torch  # local import to avoid heavy import cost when unused
        if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if pref.startswith("cuda") and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

#
def seed_everything(seed: int = 42) -> None:
    """
    Seed Python, NumPy, and PyTorch (if present). Safe to call repeatedly.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch may not be available; ignore
        pass

#
def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    """
    L2-normalize array along `axis`. Returns a new array.
    """
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)

#
def chunked(seq: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yield fixed-size chunks from a sequence."""
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

#
def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, v))

#
def require_package(name: str) -> None:
    """
    Raise a clear error if a package is missing. Useful in optional backends.
    """
    try:
        __import__(name)
    except ImportError as e:
        raise ImportError(f"Package '{name}' is required but not installed.") from e

### ========================================================================================================================================
## END (ADD IMPLEMENTATIONS IF NECESSARY)
### ========================================================================================================================================