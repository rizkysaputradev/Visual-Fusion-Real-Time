# src/models/text/registry.py
"""
Vision-Fusion-RT — Text Encoder Registry
----------------------------------------

Factory for building text encoders with a consistent interface:

Required Interface
- `.dim` : int
- `.encode_text(texts: list[str], batch_size=..., tqdm=False) -> np.ndarray [N, d]`

Available keys
- "clip_text"        → CLIP text tower (aligned with CLIP vision)
- "hf_text"          → Sentence-Transformers / Transformers fallback
- "hf_minilm"        → Preconfigured MiniLM ST model
- "hf_e5_small"      → E5-small (semantic search tuned)

Extend by adding lambdas to `TEXT_BUILDERS`.
"""

from __future__ import annotations
from typing import Callable, Dict

from .clip_text import CLIPTextEncoder
from .hf_text import HFTextEncoder


TEXT_BUILDERS: Dict[str, Callable[..., object]] = {
    # CLIP-aligned text tower (use when vision backbone is CLIP)
    "clip_text": lambda device="cpu", **kw: CLIPTextEncoder(
        model_name=kw.pop("model_name", "openai/clip-vit-base-patch32"),
        device=device,
        **kw
    ),

    # Generic semantic encoders
    "hf_text": lambda device="cpu", **kw: HFTextEncoder(
        model_name=kw.pop("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        device=device,
        **kw
    ),
    "hf_minilm": lambda device="cpu", **kw: HFTextEncoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=device,
        prefer_sentence_transformers=True,
        **kw
    ),
    "hf_e5_small": lambda device="cpu", **kw: HFTextEncoder(
        model_name="intfloat/e5-small-v2",
        device=device,
        prefer_sentence_transformers=False,  # uses transformers fallback by default
        pooling="mean",
        **kw
    ),
}


def list_text_encoders() -> list[str]:
    """Return available text encoder keys."""
    return sorted(TEXT_BUILDERS.keys())


def build_text_encoder(name: str, device: str = "cpu", **kwargs):
    """
    Construct a text encoder by registry key.

    Example
    -------
        txt = build_text_encoder("clip_text", device="cuda")
        vecs = txt.encode_text(["bottle", "red mug"])  # -> np.ndarray [2, d]
    """
    if name not in TEXT_BUILDERS:
        raise KeyError(f"Unknown text encoder '{name}'. Available: {list_text_encoders()}")
    return TEXT_BUILDERS[name](device=device, **kwargs)
