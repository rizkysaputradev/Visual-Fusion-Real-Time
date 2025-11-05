### ========================================================================================================================================
## Module       : src/eval/metrics.py
## Author       : Rizky Johan Saputra (Independent Project)
## Date         : 4th November 2025 (Seoul, South Korea)
## Project      : Vision Fusion Real Time System (Copyright 2025)
## Topics       : Computer Vision, Real-Time Systems, Interactive AI System, NLP, Machine Learning and Memory Augmentation
## Purpose      : 
## Role         : Evaluation Metrics
### ========================================================================================================================================

## ======================================================================================================
## SPECIFICATIONS
## ======================================================================================================
"""
Vision-Fusion-RT — Evaluation Metrics
-------------------------------------

This module provides common metrics for few-shot / retrieval-style recognition,
plus latency summaries for real-time evaluation.

Included
- accuracy_at_k(ranked_labels, gt_labels, k)
- cmc_curve(ranked_labels, gt_labels, k_max)
- mean_reciprocal_rank(ranked_labels, gt_labels)
- average_precision(scores, y_true) / mean_average_precision(list_of_pairs)
- roc_curve(scores, y_true)  # binary
- eer(scores, y_true)        # binary Equal Error Rate
- brier_score(probs, y_true) # calibrated probability quality
- latency_stats(lat_ms_list) # mean/median/stdev/p95/p99 + fps

Conventions
- ranked_labels: List[List[str]]  — for each query, a rank-ordered list of labels (top-1 first)
- gt_labels:     List[str]        — ground-truth label per query
- scores:        np.ndarray shape [N] or List[float] — similarity or probability for positives
- y_true:        np.ndarray shape [N] or List[int]   — 1 for positive, 0 for negative
"""

## ======================================================================================================
## SETUP (ADJUSTABLE) (ADJUST IF NECESSARY)
## ======================================================================================================
from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple, Dict, Any
import math
import numpy as np


## ======================================================================================================
## IMPLEMENTATIONS
## ======================================================================================================
#
def accuracy_at_k(ranked_labels: Sequence[Sequence[str]],
                  gt_labels: Sequence[str],
                  k: int = 1) -> float:
    """
    Top-k accuracy where a prediction is correct if the ground-truth label
    appears in the top-k ranked labels for that sample.
    """
    assert len(ranked_labels) == len(gt_labels)
    k = max(1, int(k))
    correct = 0
    for preds, y in zip(ranked_labels, gt_labels):
        preds = list(preds)[:k]
        correct += int(y in preds)
    return correct / max(1, len(gt_labels))


def mean_reciprocal_rank(ranked_labels: Sequence[Sequence[str]],
                         gt_labels: Sequence[str]) -> float:
    """
    MRR = mean( 1 / rank(y) ) over queries, where rank is 1-based.
    If y is not present, contribution is 0.
    """
    assert len(ranked_labels) == len(gt_labels)
    acc = 0.0
    for preds, y in zip(ranked_labels, gt_labels):
        try:
            r = list(preds).index(y) + 1
            acc += 1.0 / r
        except ValueError:
            pass
    return acc / max(1, len(gt_labels))


def cmc_curve(ranked_labels: Sequence[Sequence[str]],
              gt_labels: Sequence[str],
              k_max: int = 10) -> np.ndarray:
    """
    Cumulative Match Characteristic: CMC[k-1] = P(rank(y) <= k).
    """
    assert len(ranked_labels) == len(gt_labels)
    k_max = max(1, int(k_max))
    cmc = np.zeros(k_max, dtype=np.float64)
    n = len(gt_labels)

    for preds, y in zip(ranked_labels, gt_labels):
        preds = list(preds)[:k_max]
        for i in range(len(preds)):
            if preds[i] == y:
                cmc[i:] += 1.0
                break

    if n > 0:
        cmc /= n
    return cmc.astype(np.float32)


# ---------------------------------------------------------------------------
# Ranking / Detection style metrics
# ---------------------------------------------------------------------------

def average_precision(scores: Sequence[float],
                      y_true: Sequence[int]) -> float:
    """
    Average Precision for a single ranked list.

    Parameters
    ----------
    scores : list[float]
        Higher means more likely positive.
    y_true : list[int] in {0,1}
        Ground-truth binary labels, aligned with scores.

    Returns
    -------
    float in [0,1]
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int32)
    assert s.shape == y.shape

    # Sort by descending score
    order = np.argsort(-s)
    y = y[order]

    # Precision at each rank where y==1
    cum_pos = np.cumsum(y)
    ranks = np.arange(1, len(y) + 1)
    prec_at_i = cum_pos / ranks

    # AP is average of precisions at positive ranks
    if y.sum() == 0:
        return 0.0
    return float((prec_at_i[y == 1]).mean())


def mean_average_precision(list_of_pairs: Sequence[Tuple[Sequence[float], Sequence[int]]]) -> float:
    """
    mAP over multiple queries: each entry is (scores, y_true).
    """
    if len(list_of_pairs) == 0:
        return 0.0
    ap = [average_precision(scores, y) for scores, y in list_of_pairs]
    return float(np.mean(ap))


def roc_curve(scores: Sequence[float], y_true: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC (FPR, TPR, thresholds) for a set of scores and binary labels.
    Returns
    -------
    fpr, tpr, thresholds  (each np.ndarray sorted by descending threshold)
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int32)
    assert s.shape == y.shape

    # Sort descending by score
    order = np.argsort(-s)
    s = s[order]
    y = y[order]

    P = max(1, int(y.sum()))
    N = max(1, int((y == 0).sum()))

    tpr = []
    fpr = []
    thresh = []

    tp = 0
    fp = 0
    last_score = None

    for i in range(len(s)):
        if last_score is None or s[i] != last_score:
            # record point BEFORE including items with the new, lower threshold
            tpr.append(tp / P)
            fpr.append(fp / N)
            thresh.append(s[i])
            last_score = s[i]
        if y[i] == 1:
            tp += 1
        else:
            fp += 1

    # Append final point (all positives/negatives included)
    tpr.append(tp / P)
    fpr.append(fp / N)
    thresh.append(-np.inf)

    return np.asarray(fpr, dtype=np.float32), np.asarray(tpr, dtype=np.float32), np.asarray(thresh, dtype=np.float32)


def eer(scores: Sequence[float], y_true: Sequence[int]) -> float:
    """
    Equal Error Rate: point where FNR == FPR. Interpolates over ROC.
    """
    fpr, tpr, _ = roc_curve(scores, y_true)
    fnr = 1.0 - tpr
    # Find point where |FNR - FPR| is minimized
    idx = int(np.argmin(np.abs(fnr - fpr)))
    # Linear interpolation between idx and idx-1 for a bit more accuracy
    if 0 < idx < len(fpr):
        x0, y0 = fpr[idx - 1], fnr[idx - 1]
        x1, y1 = fpr[idx], fnr[idx]
        # intersection with y=x
        denom = (y1 - y0) - (x1 - x0)
        if abs(denom) > 1e-12:
            t = (x0 - y0) / denom
            e = (1 - t) * max(x0, y0) + t * max(x1, y1)
            return float(e)
    return float((fpr[idx] + fnr[idx]) / 2.0)


# ---------------------------------------------------------------------------
# Calibration metric
# ---------------------------------------------------------------------------

def brier_score(probs: Sequence[float], y_true: Sequence[int]) -> float:
    """
    Brier score for binary probabilities.
    Lower is better; perfect calibration yields smaller scores.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    assert p.shape == y.shape
    return float(np.mean((p - y) ** 2))


# ---------------------------------------------------------------------------
# Latency / throughput summaries
# ---------------------------------------------------------------------------

def latency_stats(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """
    Summarize latency distribution and derive FPS.

    Returns
    -------
    dict with keys: n, mean_ms, median_ms, stdev_ms, p95_ms, p99_ms, min_ms, max_ms, fps_mean
    """
    arr = np.asarray(list(latencies_ms), dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0, "mean_ms": 0.0, "median_ms": 0.0, "stdev_ms": 0.0,
            "p95_ms": 0.0, "p99_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "fps_mean": 0.0
        }
    mean = float(arr.mean())
    med = float(np.median(arr))
    st = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    mn = float(arr.min())
    mx = float(arr.max())
    fps = 1000.0 / mean if mean > 1e-9 else 0.0
    return {
        "n": int(arr.size),
        "mean_ms": mean,
        "median_ms": med,
        "stdev_ms": st,
        "p95_ms": p95,
        "p99_ms": p99,
        "min_ms": mn,
        "max_ms": mx,
        "fps_mean": fps,
    }

### ========================================================================================================================================
## END (ADD IMPLEMENTATIONS IF NECESSARY)
### ========================================================================================================================================