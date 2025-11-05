# src/models/text/clip_text.py
"""
Vision-Fusion-RT — CLIP Text Encoder (Hugging Face)
---------------------------------------------------

Wraps the CLIP *text* tower from HF `transformers` to produce L2-normalized
text embeddings aligned with CLIP vision features.

Interface
- `.dim` : projection dimensionality
- `.encode_text(texts: list[str], batch_size=..., tqdm=False) -> np.ndarray [N, d]`

Notes
- Uses the same model id as the paired vision tower (default: openai/clip-vit-base-patch32).
- Autocast FP16 on CUDA/MPS for speed (safe for inference).
- Handles empty input gracefully (returns shape [0, d]).
"""

from __future__ import annotations
from typing import List
import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizerFast


class CLIPTextEncoder:
    """
    Parameters
    ----------
    model_name : str
        e.g., "openai/clip-vit-base-patch32" or "...-patch16"
    device : str
        "cuda" | "mps" | "cpu"
    fp16 : bool
        Use autocast on CUDA/MPS for faster inference.
    """

    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cpu",
                 fp16: bool = True):
        self.model_name = model_name
        self.device = torch.device(device)
        self.fp16 = bool(fp16) and self.device.type in ("cuda", "mps")

        self.model = CLIPModel.from_pretrained(
            model_name,
            use_safetensors=True,
            torch_dtype=torch.float16 if self.fp16 else torch.float32,
        )


        self.model.eval().to(self.device)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)

        self.dim = int(self.model.config.projection_dim)

    @torch.inference_mode()
    def encode_text(self,
                    texts: List[str],
                    batch_size: int = 64,
                    tqdm: bool = False) -> np.ndarray:
        """
        Encode a list of strings into CLIP text embeddings (L2-normalized).

        Returns
        -------
        np.ndarray
            Float32 array of shape [N, d].
        """
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype="float32")

        iter_range = range(0, len(texts), batch_size)
        if tqdm:
            try:
                from tqdm import tqdm as _tqdm
                iter_range = _tqdm(iter_range, desc="encode_text[clip]")
            except Exception:
                pass

        out_chunks = []
        cuda_autocast = torch.cuda.amp.autocast if (self.fp16 and self.device.type == "cuda") else None
        mps_autocast = torch.autocast if (self.fp16 and self.device.type == "mps") else None

        for i in iter_range:
            chunk = texts[i:i + batch_size]
            tokens = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            if cuda_autocast is not None:
                with cuda_autocast():
                    feats = self.model.get_text_features(**tokens)
            elif mps_autocast is not None:
                with mps_autocast(device_type="mps"):
                    feats = self.model.get_text_features(**tokens)
            else:
                feats = self.model.get_text_features(**tokens)

            feats = torch.nn.functional.normalize(feats, dim=-1)
            out_chunks.append(feats.detach().cpu().to(torch.float32).numpy())

        return np.concatenate(out_chunks, axis=0)
