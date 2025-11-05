# src/models/backbones/clip_vision.py
"""
Vision-Fusion-RT — CLIP Vision Backbone (Hugging Face)
------------------------------------------------------

This module wraps the CLIP *vision* tower from Hugging Face transformers to
produce L2-normalized image embeddings suitable for cosine / inner-product
retrieval. It provides:

- `.dim` — projection dimensionality (d)
- `.encode_images(imgs, batch_size=..., tqdm=False)` → np.ndarray [N, d]
- Robust PIL/ndarray handling (OpenCV BGR or PIL RGB)
- Optional FP16 autocast on CUDA/MPS for throughput
- Stateless preprocessing via `CLIPProcessor`

Dependencies
- transformers >= 4.30
- torch
- pillow
- numpy
"""

from __future__ import annotations
from typing import Iterable, List, Optional, Union
import numpy as np
from PIL import Image

import torch
from transformers import CLIPModel, CLIPProcessor


ArrayLike = Union[np.ndarray, Image.Image]


class CLIPVisionEncoder:
    """
    Parameters
    ----------
    model_name : str
        HF model id for CLIP, e.g., "openai/clip-vit-base-patch32" or "openai/clip-vit-base-patch16".
    device : str
        "cuda" | "mps" | "cpu"
    fp16 : bool
        Use autocast half precision on CUDA/MPS for speed (kept false on CPU).
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cpu",
                 fp16: bool = True):
        self.model_name = model_name
        self.device = torch.device(device)
        self.fp16 = bool(fp16) and self.device.type in ("cuda", "mps")

        # Lazy-load model & processor
        self.model = CLIPModel.from_pretrained(
            model_name,
            use_safetensors=True,                         # <-- avoid torch.load on .bin
            torch_dtype=torch.float16 if self.fp16 else torch.float32,
        )
        self.model.eval().to(self.device)

        self.processor = CLIPProcessor.from_pretrained(
            model_name,
            use_safetensors=True,
        )

        # Embedding dimension (projection head)
        self.dim = int(self.model.config.projection_dim)

    # -------------------------
    # Utilities
    # -------------------------

    @staticmethod
    def _to_pil(x: ArrayLike) -> Image.Image:
        """Convert ndarray (BGR) or PIL → PIL RGB."""
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        # assume OpenCV BGR HxWx3
        return Image.fromarray(x[:, :, ::-1])

    # -------------------------
    # Encoding
    # -------------------------

    @torch.inference_mode()
    def encode_images(self,
                      imgs: List[ArrayLike],
                      batch_size: int = 32,
                      tqdm: bool = False) -> np.ndarray:
        """
        Encode a list of images into L2-normalized embeddings in CLIP space.

        Returns
        -------
        np.ndarray
            Float32 array of shape [N, d] with unit L2 norm.
        """
        if len(imgs) == 0:
            return np.zeros((0, self.dim), dtype="float32")

        iter_range = range(0, len(imgs), batch_size)
        if tqdm:
            try:
                from tqdm import tqdm as _tqdm
                iter_range = _tqdm(iter_range, desc="encode_images")
            except Exception:
                pass

        out_chunks = []
        # Choose autocast context depending on device
        cuda_autocast = torch.cuda.amp.autocast if (self.fp16 and self.device.type == "cuda") else None
        mps_autocast = torch.autocast if (self.fp16 and self.device.type == "mps") else None

        for i in iter_range:
            batch = [self._to_pil(im) for im in imgs[i:i + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if cuda_autocast is not None:
                with cuda_autocast():
                    feats = self.model.get_image_features(**inputs)
            elif mps_autocast is not None:
                with mps_autocast(device_type="mps"):
                    feats = self.model.get_image_features(**inputs)
            else:
                feats = self.model.get_image_features(**inputs)

            # L2 normalize
            feats = torch.nn.functional.normalize(feats, dim=-1)
            out_chunks.append(feats.detach().cpu().to(torch.float32).numpy())

        return np.concatenate(out_chunks, axis=0)
