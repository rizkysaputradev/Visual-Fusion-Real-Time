# src/models/backbones/registry.py
"""
Vision-Fusion-RT — Backbone Registry
------------------------------------

Factory for building image encoders that expose a *common* interface:

Required Interface
- `.dim` : int               — embedding dimensionality
- `.encode_images(imgs, batch_size=...) -> np.ndarray [N, d]` (L2-normalized)
- (optional) `preproc()`     — callable that returns a preprocessing function

Available keys
- CLIP (HF):  "clip_vit_b32", "clip_vit_b16"
- ViT (timm): "vit_b16_timm", "vit_b32_timm", "vit_small16_timm", "vit_large16_timm"

Add new encoders by extending BACKBONE_BUILDERS with a lambda.
"""

from __future__ import annotations
from typing import Callable, Dict

from .clip_vision import CLIPVisionEncoder
from .vit_vision import TimmViTEncoder


BACKBONE_BUILDERS: Dict[str, Callable[..., object]] = {
    # --- CLIP (HF) ---
    "clip_vit_b32": lambda device="cpu", **kw: CLIPVisionEncoder(
        model_name="openai/clip-vit-base-patch32", device=device, **kw
    ),
    "clip_vit_b16": lambda device="cpu", **kw: CLIPVisionEncoder(
        model_name="openai/clip-vit-base-patch16", device=device, **kw
    ),

    # --- ViT (timm) ---
    "vit_b16_timm": lambda device="cpu", **kw: TimmViTEncoder(
        model_name="vit_base_patch16_224", device=device, **kw
    ),
    "vit_b32_timm": lambda device="cpu", **kw: TimmViTEncoder(
        model_name="vit_base_patch32_224", device=device, **kw
    ),
    "vit_small16_timm": lambda device="cpu", **kw: TimmViTEncoder(
        model_name="vit_small_patch16_224", device=device, **kw
    ),
    "vit_large16_timm": lambda device="cpu", **kw: TimmViTEncoder(
        model_name="vit_large_patch16_224", device=device, **kw
    ),
}


def list_backbones() -> list[str]:
    """Return available registry keys (sorted)."""
    return sorted(BACKBONE_BUILDERS.keys())


def build_backbone(name: str, device: str = "cpu", **kwargs):
    """
    Construct a backbone by registry key.

    Example
    -------
        enc = build_backbone("clip_vit_b32", device="cuda")
        vec = enc.encode_images([pil])  # -> np.ndarray [1, d]
    """
    if name not in BACKBONE_BUILDERS:
        raise KeyError(f"Unknown backbone '{name}'. Available: {list_backbones()}")
    return BACKBONE_BUILDERS[name](device=device, **kwargs)
