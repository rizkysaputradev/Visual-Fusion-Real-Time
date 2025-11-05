# src/models/preproc/transforms.py
"""
Vision-Fusion-RT — Preprocessing Transforms
-------------------------------------------

Purpose
- Provide consistent, reusable image transforms for encoders that *do not*
  expose their own processors (e.g., some timm or custom backbones).
- Export small utilities for PIL/ndarray conversion and normalization.

Design
- We keep transforms torch/torchvision-based to leverage GPU pipelines when needed.
- Presets ("clip", "imagenet") reflect common normalization conventions.
- Every transform returned by `build_image_transform(...)` expects a PIL Image (RGB)
  and outputs a float32 CHW Tensor in range [-something, +something] depending on preset.

Notes
- CLIP encoders in our repo use their own CLIPProcessor; you rarely need these for CLIP.
- ViT (timm) backbones already build self.transforms from default_cfg. These utilities
  are primarily for custom or experimental backbones, offline preprocessing, and data tools.
"""

from __future__ import annotations
from typing import Callable, Iterable, Tuple, Literal, Optional
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T


# -------------------------
# Preset builders
# -------------------------

_PRESETS = ("clip", "imagenet", "unit", "raw")


def _clip_norm() -> T.Normalize:
    # CLIP normalization uses mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
    return T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                       std=(0.26862954, 0.26130258, 0.27577711))


def _imagenet_norm() -> T.Normalize:
    # Classic ImageNet normalization
    return T.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225))


def _unit_norm() -> Callable[[torch.Tensor], torch.Tensor]:
    # Map from [0,1] to [-1,1]
    class Unit:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return x.mul_(2.0).add_(-1.0)
    return Unit()


def _identity() -> Callable[[torch.Tensor], torch.Tensor]:
    class Identity:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return x
    return Identity()


def _resize_center_crop(size: Tuple[int, int], center_crop: bool = True) -> T.Compose:
    w, h = int(size[0]), int(size[1])
    ops = [T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC)]
    if center_crop:
        ops.append(T.CenterCrop((h, w)))
    return T.Compose(ops)


def build_image_transform(
    size: Tuple[int, int] = (224, 224),
    preset: Literal["clip", "imagenet", "unit", "raw"] = "clip",
    center_crop: bool = False,
) -> T.Compose:
    """
    Build a PIL->Tensor transform.
    - Input: PIL Image (RGB)
    - Output: torch.FloatTensor [3, H, W]
    """
    normalizer = {
        "clip": _clip_norm(),
        "imagenet": _imagenet_norm(),
        "unit": _unit_norm(),
        "raw": _identity(),
    }[preset]

    return T.Compose([
        _resize_center_crop(size, center_crop=center_crop),
        T.ToTensor(),           # [0,1]
        normalizer,             # normalization per preset
    ])


# -------------------------
# Utilities
# -------------------------

def pil_from_bgr(bgr: np.ndarray) -> Image.Image:
    """OpenCV BGR ndarray -> PIL RGB Image."""
    from cv2 import cvtColor, COLOR_BGR2RGB
    rgb = cvtColor(bgr, COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def bgr_from_pil(pil: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR ndarray."""
    import cv2
    arr = np.array(pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
