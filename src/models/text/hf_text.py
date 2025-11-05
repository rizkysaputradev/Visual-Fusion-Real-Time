# src/models/text/hf_text.py
"""
Vision-Fusion-RT — HF/Sentence-Transformers Text Encoder (Fallback/Generic)
---------------------------------------------------------------------------

Provides a flexible text encoder using either:
1) `sentence-transformers` (preferred: stable pooling & normalization), or
2) vanilla `transformers` (CLS token / mean pooling) when ST is unavailable.

Goal
- Offer a general-purpose semantic embedding space for labels/prompts when CLIP
  text is not desired, while still keeping a consistent interface.

Interface
- `.dim` : output dimensionality
- `.encode_text(texts: list[str], batch_size=..., tqdm=False) -> np.ndarray [N, d]`

Defaults
- Sentence-Transformers: "sentence-transformers/all-MiniLM-L6-v2"
- Transformers fallback: "sentence-transformers/all-MiniLM-L6-v2" tokenizer + model via HF
"""

from __future__ import annotations
from typing import List, Literal, Optional
import numpy as np
import torch

# Try to import sentence-transformers; if missing, use transformers fallback.
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

from transformers import AutoModel, AutoTokenizer


def _mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Standard mean pooling used by Sentence-Transformers."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


class HFTextEncoder:
    """
    Parameters
    ----------
    model_name : str
        e.g., "sentence-transformers/all-MiniLM-L6-v2"
    device : str
        "cuda" | "mps" | "cpu"
    fp16 : bool
        Autocast half precision on CUDA/MPS for throughput.
    pooling : {"mean", "cls"}
        Pooling strategy for transformers fallback (ignored for ST models).
    normalize : bool
        L2-normalize outputs.
    prefer_sentence_transformers : bool
        If True and ST is installed, use it; else transformers fallback.
    """

    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu",
                 fp16: bool = True,
                 pooling: Literal["mean", "cls"] = "mean",
                 normalize: bool = True,
                 prefer_sentence_transformers: bool = True):
        self.model_name = model_name
        self.device = torch.device(device)
        self.fp16 = bool(fp16) and self.device.type in ("cuda", "mps")
        self.pooling = pooling
        self.normalize = normalize

        self.using_st = bool(prefer_sentence_transformers and _HAS_ST)
        if self.using_st:
            # Sentence-Transformers provides embedding dimension at runtime
            self.model = SentenceTransformer(model_name, device=str(self.device))
            self.dim = int(self.model.get_sentence_embedding_dimension())
        else:
            # Transformers fallback (AutoModel + AutoTokenizer)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
            # Determine dim by a dummy forward
            with torch.inference_mode():
                tok = self.tokenizer(["dummy"], return_tensors="pt", padding=True, truncation=True).to(self.device)
                out = self.model(**tok)
                if self.pooling == "cls":
                    d = int(out.last_hidden_state[:, 0].shape[-1])
                else:
                    d = int(out.last_hidden_state.shape[-1])
            self.dim = d

    @torch.inference_mode()
    def encode_text(self,
                    texts: List[str],
                    batch_size: int = 64,
                    tqdm: bool = False) -> np.ndarray:
        """
        Encode list of strings into semantic embeddings. L2-normalized if `normalize=True`.
        """
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype="float32")

        if self.using_st:
            # Sentence-Transformers path (already batched internally)
            emb = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=tqdm,
                normalize_embeddings=self.normalize,
            )
            return emb.astype("float32")

        # Transformers fallback path
        iter_range = range(0, len(texts), batch_size)
        if tqdm:
            try:
                from tqdm import tqdm as _tqdm
                iter_range = _tqdm(iter_range, desc=f"encode_text[{self.model_name}]")
            except Exception:
                pass

        chunks = []
        cuda_autocast = torch.cuda.amp.autocast if (self.fp16 and self.device.type == "cuda") else None
        mps_autocast = torch.autocast if (self.fp16 and self.device.type == "mps") else None

        for i in iter_range:
            batch = texts[i:i + batch_size]
            tok = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)

            if cuda_autocast is not None:
                with cuda_autocast():
                    out = self.model(**tok)
            elif mps_autocast is not None:
                with mps_autocast(device_type="mps"):
                    out = self.model(**tok)
            else:
                out = self.model(**tok)

            if self.pooling == "cls":
                feats = out.last_hidden_state[:, 0]              # [B, H]
            else:
                feats = _mean_pooling(out.last_hidden_state, tok["attention_mask"])

            if self.normalize:
                feats = torch.nn.functional.normalize(feats, dim=-1)
            chunks.append(feats.detach().cpu().to(torch.float32).numpy())

        return np.concatenate(chunks, axis=0)
