# src/models/preproc/augment.py
"""
Vision-Fusion-RT — Light Augmentations for Few-Shot Registration
----------------------------------------------------------------

Motivation
- For *few-shot* incremental registration, mild, label-preserving augmentations
  can improve robustness without re-training a full model. We avoid heavy
  distortions that would shift semantics (e.g., extreme color changes).

Features
- Augmenter class with configurable "strength" (low/mid/high) presets.
- Deterministic option via seeding (useful for reproducible experiments).
- Outputs PIL Images (RGB) suitable for encoder preprocessors.

Augmentations (toggle by strength)
- RandomResizedCrop (mild; near center bias)
- HorizontalFlip (p=0.1..0.3)
- ColorJitter (small brightness/contrast/saturation jitter)
- RandomGrayscale (p up to 0.2)
- GaussianBlur (σ small, p up to 0.2)

Usage
    aug = Augmenter(strength="mid", seed=42)
    variants = aug.apply([pil_img1, pil_img2], n_per_image=4)
"""

from __future__ import annotations
from typing import Iterable, List, Literal
import random

import torchvision.transforms as T
from PIL import Image


class Augmenter:
    def __init__(self, strength: Literal["low", "mid", "high"] = "mid", seed: int | None = None):
        self.strength = strength
        if seed is not None:
            random.seed(seed)

        if strength == "low":
            self._p_flip = 0.10
            self._p_gray = 0.05
            self._p_blur = 0.05
            self._jitter = (0.05, 0.05, 0.05, 0.02)
            self._scale = (0.9, 1.0)
        elif strength == "high":
            self._p_flip = 0.30
            self._p_gray = 0.20
            self._p_blur = 0.20
            self._jitter = (0.20, 0.20, 0.20, 0.05)
            self._scale = (0.7, 1.0)
        else:  # "mid"
            self._p_flip = 0.20
            self._p_gray = 0.10
            self._p_blur = 0.10
            self._jitter = (0.10, 0.10, 0.10, 0.03)
            self._scale = (0.8, 1.0)

        # Mild center-preserving crop preference:
        self._crop = T.RandomResizedCrop(
            size=224,  # final size is usually controlled by encoder transform; crop is approximate
            scale=self._scale,
            ratio=(0.9, 1.1),
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        )

        blur_kernel = 3
        self.pipeline = T.Compose([
            self._crop,
            T.RandomHorizontalFlip(p=self._p_flip),
            T.ColorJitter(*self._jitter),
            T.RandomGrayscale(p=self._p_gray),
            T.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 1.0), p=self._p_blur),
        ])

    def apply(self, images: Iterable[Image.Image], n_per_image: int = 4) -> List[Image.Image]:
        """
        Generate augmented variants for each input PIL Image.

        Returns
        -------
        List[PIL.Image.Image]
            A flat list length = len(images) * n_per_image
        """
        out: List[Image.Image] = []
        for im in images:
            im = im.convert("RGB")
            for _ in range(n_per_image):
                out.append(self.pipeline(im))
        return out
