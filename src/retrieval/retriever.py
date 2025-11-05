# src/retrieval/retriever.py
"""
Vision-Fusion-RT — Retriever
----------------------------

This module implements the high-level retrieval pipeline that:
1) Encodes an input image with the configured vision backbone.
2) Queries the vector store (FAISS/Milvus) for k-NN/ANN neighbors.
3) Aggregates neighbor votes into per-label scores.
4) Optionally fuses with text-prototype priors (CLIP text).
5) Applies temporal smoothing (EMA) for stable streaming output.

It is intentionally *decision-head agnostic*: this class returns the ranked
labels + raw fused scores; your `pipeline/inference.py` layer can feed those
into `heads/decision.py` (temperature + open-set) to finalize the label.

Contracts
- Vision encoder exposes `.encode_images([BGR ndarray or PIL], batch_size) -> [N, d]`
- Text encoder (optional) exposes `.encode_text([str]) -> [M, d]` (aligned space)
- Memory manager exposes `.store.search(q, k)` and `.registry.meta_of(vid)`.

Notes
- If your text encoder is not aligned to the vision space (e.g., MiniLM), set
  `text_aligned=False` in the constructor to disable image↔text fusion.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.core.types import RetrievalResult
from src.core.logging_utils import FPSMeter
from src.io.image_io import to_pil
from src.memory.incremental import IncrementalMemory
from . import fusion as F


@dataclass
class RetrieverConfig:
    k: int = 5
    alpha_fusion: float = 0.7
    temporal_ema: float = 0.2                  # 0.0 disables smoothing
    neighbor_agg: str = "max"                  # "max" | "mean" | "sum"
    text_prompt_template: str = "{label}"      # used when encoding text prototypes


class Retriever:
    """
    End-to-end retriever with optional text fusion and temporal smoothing.

    Attributes
    ----------
    memory : IncrementalMemory
    img_enc : image encoder (backbone wrapper)
    txt_enc : optional text encoder (aligned to image space)
    """

    def __init__(self,
                 memory: IncrementalMemory,
                 image_encoder,
                 text_encoder=None,
                 text_aligned: bool = True,
                 cfg: RetrieverConfig = RetrieverConfig()):
        self.memory = memory
        self.img_enc = image_encoder
        self.txt_enc = text_encoder
        self.text_aligned = bool(text_aligned)
        self.cfg = cfg

        # Text prototypes cache: label -> vector
        self._text_proto: Dict[str, np.ndarray] = {}
        # Temporal EMA cache: label -> prob (applied on top-1 stream)
        self._ema_prev: Optional[Tuple[str, float]] = None

    # ---------------------------------------------------------------------
    # Text prototypes
    # ---------------------------------------------------------------------

    def set_text_prototypes(self, label_to_prompt: Dict[str, str]) -> None:
        """
        Encode text prompts per label and cache them as normalized vectors.
        """
        if self.txt_enc is None or not self.text_aligned:
            self._text_proto.clear()
            return

        labels = list(label_to_prompt.keys())
        prompts = [label_to_prompt[lbl] for lbl in labels]
        vecs = self.txt_enc.encode_text(prompts, batch_size=max(8, len(prompts)))
        # Ensure float32, L2-normalized (encoders should already do this)
        vecs = np.asarray(vecs, dtype=np.float32)
        # Map back into dict
        self._text_proto = {lbl: vecs[i] for i, lbl in enumerate(labels)}

    def update_text_prototypes(self, labels: Sequence[str]) -> None:
        """
        Convenience: build prompts from template and set prototypes.
        """
        mapping = {lbl: self.cfg.text_prompt_template.format(label=lbl) for lbl in labels}
        self.set_text_prototypes(mapping)

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------

    def encode_image(self, bgr: np.ndarray) -> np.ndarray:
        """
        Encode one image into a single embedding vector [d].
        """
        vec = self.img_enc.encode_images([bgr], batch_size=1)
        return vec[0]  # [d]

    def _neighbors(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
        """
        Search the store and resolve neighbor IDs to labels.
        Returns
        -------
        sims : [1, k] similarities
        ids  : [1, k] vector ids
        lbls : [[str]*k]
        """
        D, I = self.memory.store.search(q[None, :], k=k)  # [1, k]
        sims = F.as_similarity(D, metric=self.memory.state.metric)  # [1, k]
        lbls: List[List[str]] = [[]]
        for j in range(I.shape[1]):
            vid = int(I[0, j])
            meta = self.memory.registry.meta_of(vid)
            lbls[0].append(meta.label if meta is not None else "")
        return sims, I, lbls

    def _text_scores_for_labels(self, labels: Sequence[str], img_vec: np.ndarray) -> Dict[str, float]:
        """
        Compute image↔text similarities for the given labels (only if aligned).
        """
        if self.txt_enc is None or not self.text_aligned or not self._text_proto:
            return {}
        # Stack label prototypes (only those present)
        keys = [lbl for lbl in labels if lbl in self._text_proto]
        if not keys:
            return {}
        mat = np.stack([self._text_proto[lbl] for lbl in keys], axis=0).astype(np.float32)  # [M, d]
        # cosine/IP similarity (embeddings are normalized)
        sims = mat @ img_vec.astype(np.float32)  # [M]
        return {lbl: float(sims[i]) for i, lbl in enumerate(keys)}

    def retrieve(self, bgr: np.ndarray) -> RetrievalResult:
        """
        Full retrieval for one frame.
        - Encodes image
        - Searches neighbors
        - Aggregates label scores
        - Optional text fusion
        - Temporal smoothing on the top-1 probability (downstream can apply decision head)

        Returns
        -------
        RetrievalResult(label, score, neighbors, latency_ms)
            neighbors: list of (label, sim, id) for the raw k-NN
        """
        t0 = FPSMeter()._last  # not used; we'll compute simple latency

        # 1) Encode
        q = self.encode_image(bgr)  # [d]

        # 2) k-NN
        sims, ids, lbls = self._neighbors(q, k=self.cfg.k)  # sims: [1, k]
        sims = sims[0]
        ids = ids[0]
        lbls0 = lbls[0]

        # 3) Aggregate neighbor votes → per-label scores
        scores_list = F.aggregate_neighbors_to_labels([lbls0], sims[None, :], strategy=self.cfg.neighbor_agg)
        label_scores = scores_list[0]  # dict

        # 4) Text fusion (if aligned)
        text_scores = self._text_scores_for_labels(list(label_scores.keys()), img_vec=q)
        fused = F.fuse_label_scores_with_text(label_scores, text_scores, alpha=self.cfg.alpha_fusion)

        # 5) Rank
        labels_sorted, scores_sorted = F.topk_from_label_scores(fused, k=max(1, self.cfg.k))
        if len(labels_sorted) == 0:
            return RetrievalResult(label="unknown", score=0.0, neighbors=[], latency_ms=0.0)

        # Build neighbor triples for inspection
        neighbors = [(lbls0[j], float(sims[j]), int(ids[j])) for j in range(len(lbls0))]

        # Temporal EMA smoothing on *top-1* (optional)
        top1_label = labels_sorted[0]
        top1_score = float(scores_sorted[0])
        if self.cfg.temporal_ema > 0.0:
            if self._ema_prev is None or self._ema_prev[0] != top1_label:
                ema_score = top1_score
            else:
                ema_prev = self._ema_prev[1]
                beta = float(self.cfg.temporal_ema)
                ema_score = beta * ema_prev + (1.0 - beta) * top1_score
            self._ema_prev = (top1_label, float(ema_score))
            top1_score = float(ema_score)

        # Simple latency estimate (wall clock from encode→here). If you want exact,
        # wrap with PerfAccumulator in pipeline/inference.py.
        latency_ms = 0.0  # placeholder; measured upstream

        return RetrievalResult(
            label=top1_label,
            score=top1_score,
            neighbors=neighbors,
            latency_ms=latency_ms,
        )

    # ---------------------------------------------------------------------
    # Batch API (optional, useful for offline eval)
    # ---------------------------------------------------------------------

    def retrieve_batch(self, batch_bgr: List[np.ndarray]) -> List[RetrievalResult]:
        out: List[RetrievalResult] = []
        # Batch encode for speed
        vecs = self.img_enc.encode_images(batch_bgr, batch_size=max(4, len(batch_bgr)))
        for i, q in enumerate(vecs):
            sims, ids, lbls = self._neighbors(q, k=self.cfg.k)
            sims = sims[0]
            ids = ids[0]
            lbls0 = lbls[0]
            scores_list = F.aggregate_neighbors_to_labels([lbls0], sims[None, :], strategy=self.cfg.neighbor_agg)
            fused = scores_list[0]

            # optional text fusion with current query vector
            text_scores = self._text_scores_for_labels(list(fused.keys()), img_vec=q)
            fused = F.fuse_label_scores_with_text(fused, text_scores, alpha=self.cfg.alpha_fusion)

            labels_sorted, scores_sorted = F.topk_from_label_scores(fused, k=max(1, self.cfg.k))
            if len(labels_sorted) == 0:
                out.append(RetrievalResult(label="unknown", score=0.0, neighbors=[], latency_ms=0.0))
                continue

            neighbors = [(lbls0[j], float(sims[j]), int(ids[j])) for j in range(len(lbls0))]
            out.append(RetrievalResult(
                label=labels_sorted[0],
                score=float(scores_sorted[0]),
                neighbors=neighbors,
                latency_ms=0.0,
            ))
        return out
