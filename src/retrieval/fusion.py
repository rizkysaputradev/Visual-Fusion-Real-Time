# src/retrieval/fusion.py
"""
Vision-Fusion-RT — Retrieval Fusion Utilities
---------------------------------------------

This module centralizes the math used to combine:
1) Per-label scores from ANN neighbors (image side), and
2) Optional text priors/prototypes (text side, e.g., CLIP text).

Key ideas
- Our image encoders output L2-normalized embeddings; when using FAISS/IP, the
  "distance" returned is actually a similarity (dot product == cosine).
- For L2 metrics, we transform distances to similarities via `sim = -dist`
  (higher is better), which is consistent for ranking/thresholding.

- We allow different neighbor-to-label aggregation strategies:
    * "max"  : strongest vote wins (robust under noisy labels)
    * "mean" : average vote (smooth for multi-instance per class)
    * "sum"  : sum of votes (biases frequent labels; useful in some setups)

- Late fusion of modalities uses a convex combination:
    s_fused[label] = alpha * s_img[label] + (1 - alpha) * s_txt[label]
  with automatic handling if text prior missing.

Everything here is **framework-agnostic** (NumPy only).
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import numpy as np


# -----------------------------
# Distance/Similarity helpers
# -----------------------------

def as_similarity(D: np.ndarray, metric: str = "ip") -> np.ndarray:
    """
    Convert FAISS/Milvus output distances into *similarities* (higher is better).
    - For inner product ("ip"): already similarity, return as-is.
    - For L2 ("l2"): negate distance.

    Parameters
    ----------
    D : np.ndarray [N, k]
        Raw distance/similarity matrix from backend.
    metric : "ip" | "l2"
    """
    D = np.asarray(D, dtype=np.float32)
    if metric == "ip":
        return D
    # L2: smaller distance = better; turn into similarity by negation
    return -D


# -----------------------------
# Label aggregation
# -----------------------------

def aggregate_neighbors_to_labels(
    neighbor_labels: Sequence[Sequence[str]],
    neighbor_sims: np.ndarray,  # [N, k] similarities (higher=better)
    strategy: str = "max",
) -> List[Dict[str, float]]:
    """
    Aggregate neighbor similarities into per-label scores for each query.

    Returns
    -------
    per_query_scores : List[Dict[label, score]]
        For each query i, a dict mapping label -> aggregated score.
    """
    strategy = strategy.lower()
    if strategy not in ("max", "mean", "sum"):
        raise ValueError("strategy must be 'max' | 'mean' | 'sum'")

    N, k = neighbor_sims.shape
    out: List[Dict[str, float]] = []

    for i in range(N):
        agg: Dict[str, List[float]] = {}
        for j in range(k):
            lbl = neighbor_labels[i][j]
            if lbl is None or lbl == "":
                continue
            s = float(neighbor_sims[i, j])
            agg.setdefault(lbl, []).append(s)

        # reduce
        red: Dict[str, float] = {}
        for lbl, vals in agg.items():
            if strategy == "max":
                red[lbl] = float(np.max(vals))
            elif strategy == "mean":
                red[lbl] = float(np.mean(vals))
            else:  # "sum"
                red[lbl] = float(np.sum(vals))
        out.append(red)

    return out


# -----------------------------
# Text fusion
# -----------------------------

def fuse_label_scores_with_text(
    label_scores: Dict[str, float],
    text_scores: Mapping[str, float] | None,
    alpha: float = 0.7,
) -> Dict[str, float]:
    """
    Late fuse image-derived label scores with optional text priors/prototypes.

    If `text_scores` is None or missing a label, we fall back to image only.

    Parameters
    ----------
    label_scores : Dict[label, score]
        Scores from neighbors (already similarities).
    text_scores : Dict[label, score] | None
        Text prior per label (e.g., image-vs-text similarity, or prior weight).
    alpha : float
        Blend weight for image (1.0=image only; 0.0=text only).
    """
    if not text_scores:
        return dict(label_scores)

    a = float(alpha)
    out: Dict[str, float] = {}
    for lbl, s_img in label_scores.items():
        s_txt = float(text_scores.get(lbl, s_img))
        out[lbl] = a * float(s_img) + (1.0 - a) * s_txt
    return out


# -----------------------------
# Post-processing / ranking
# -----------------------------

def topk_from_label_scores(label_scores: Dict[str, float], k: int = 5) -> Tuple[List[str], np.ndarray]:
    """
    Return the top-k labels and scores (sorted desc).
    """
    if not label_scores:
        return [], np.zeros((0,), dtype=np.float32)
    items = sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True)[: int(k)]
    labels = [it[0] for it in items]
    scores = np.array([it[1] for it in items], dtype=np.float32)
    return labels, scores
