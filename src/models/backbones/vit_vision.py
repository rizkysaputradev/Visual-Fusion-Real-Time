# src/models/backbones/vit_vision.py
"""
Vision-Fusion-RT — ViT Backbone (timm)
---------------------------------------

This module wraps popular Vision Transformer backbones from `timm` as *feature
extractors* (num_classes=0), yielding L2-normalized embeddings suitable for
retrieval.

- `.dim` — output feature dimensionality
- `.encode_images(imgs, batch_size=..., tqdm=False)` → np.ndarray [N, d]
- Preprocessing derived from model.default_cfg (resize/normalize)
- Autocast FP16 on CUDA/MPS when requested

Useful model names
- "vit_base_patch16_224"   (good balance)
- "vit_base_patch32_224"
- "vit_small_patch16_224"
- "vit_large_patch16_224"

Dependencies
- timm
- torch
- torchvision
- pillow
- numpy
"""

from __future__ import annotations
from typing import List, Union
import numpy as np
from PIL import Image

import torch
import timm
import torchvision.transforms as T


ArrayLike = Union[np.ndarray, Image.Image]


class TimmViTEncoder:
    """
    Parameters
    ----------
    model_name : str
        timm model id, e.g., "vit_base_patch16_224".
    device : str
        "cuda" | "mps" | "cpu"
    fp16 : bool
        Autocast half precision on CUDA/MPS for speed.
    """

    def __init__(self,
                 model_name: str = "vit_base_patch16_224",
                 device: str = "cpu",
                 fp16: bool = True):
        self.model_name = model_name
        self.device = torch.device(device)
        self.fp16 = bool(fp16) and self.device.type in ("cuda", "mps")

        # Feature extractor (num_classes=0 returns penultimate features)
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval().to(self.device)

        # Build preprocessing from timm's default_cfg
        dc = self.model.default_cfg
        img_size = dc.get("input_size", (3, 224, 224))[-1]
        mean = dc.get("mean", (0.5, 0.5, 0.5))
        std = dc.get("std", (0.5, 0.5, 0.5))

        self.transforms = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        # Infer output dimensionality with a dummy forward
        with torch.inference_mode():
            x = torch.zeros(1, 3, img_size, img_size, device=self.device)
            y = self.model(x)
            self.dim = int(y.shape[-1])

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
        Encode a list of images into L2-normalized embeddings.

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
                iter_range = _tqdm(iter_range, desc=f"encode_images[{self.model_name}]")
            except Exception:
                pass

        out_chunks = []
        cuda_autocast = torch.cuda.amp.autocast if (self.fp16 and self.device.type == "cuda") else None
        mps_autocast = torch.autocast if (self.fp16 and self.device.type == "mps") else None

        for i in iter_range:
            chunk = [self._to_pil(im) for im in imgs[i:i + batch_size]]
            tensor = torch.stack([self.transforms(im) for im in chunk], dim=0).to(self.device)

            if cuda_autocast is not None:
                with cuda_autocast():
                    feats = self.model(tensor)
            elif mps_autocast is not None:
                with mps_autocast(device_type="mps"):
                    feats = self.model(tensor)
            else:
                feats = self.model(tensor)

            feats = torch.nn.functional.normalize(feats, dim=-1)
            out_chunks.append(feats.detach().cpu().to(torch.float32).numpy())

        return np.concatenate(out_chunks, axis=0)
