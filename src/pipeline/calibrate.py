# src/pipeline/calibrate.py
"""
Vision-Fusion-RT — Calibration Utilities
----------------------------------------

This module provides practical calibration helpers to tune:
1) Temperature (DecisionHead.temperature) on a small *known-class* dev set
   using negative log-likelihood over the top-k fused scores returned by the retriever.
2) Open-set threshold tau (DecisionHead.tau_open) using a held-out *unknown* set.

Design notes
- We do not change the retriever; we treat its fused per-label scores as “logits”.
  Since Retriever currently exposes only top-1 publicly, we compute per-sample
  top-k lists by re-invoking its internal math. To keep this module decoupled,
  we use the public `.retrieve_batch` and consider the **top-k** scores as logits.
  Samples where the ground truth is not in the top-k are skipped for temperature fit.

APIs
- fit_temperature_on_dev(retriever, decision, images_bgr, labels, k=5) -> float
- choose_openset_threshold(retriever, decision, known_imgs, unknown_imgs, k=5, target_fpr=0.05) -> float
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

from src.retrieval.retriever import Retriever
from src.models.heads.decision import DecisionHead
from src.pipeline.encode import ensure_l2


def fit_temperature_on_dev(
    retriever: Retriever,
    decision: DecisionHead,
    images_bgr: List[np.ndarray],
    labels: List[str],
    k: int = 5,
) -> float:
    """
    Fit temperature to minimize NLL on a small dev set.
    We approximate logits with the top-k fused scores from Retriever (sorted).
    If the ground-truth label is not in top-k, that sample is skipped.
    """
    assert len(images_bgr) == len(labels)
    logits_list: List[np.ndarray] = []
    targets: List[int] = []

    # Batch in moderate chunks to avoid starving GPU
    B = 16
    for i in range(0, len(images_bgr), B):
        chunk = images_bgr[i:i + B]
        rets = retriever.retrieve_batch(chunk)
        for ret, y in zip(rets, labels[i:i + B]):
            # Build a pseudo top-k list from neighbors by reducing to label scores:
            # We already have the top-1 from Retriever; but for temperature fitting we
            # need at least two logits. Approximate with (top1, top2=mean of remaining).
            # If you want precise per-label vectors, extend Retriever to expose them.
            if len(ret.neighbors) == 0:
                continue
            # Group neighbor sims by label
            label_scores = {}
            for lbl, sim, _vid in ret.neighbors:
                if not lbl:
                    continue
                label_scores.setdefault(lbl, []).append(float(sim))
            label_scores = {k: float(np.mean(v)) for k, v in label_scores.items()}
            # Sort and keep top-k
            items = sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True)[:max(2, k)]
            labs = [t[0] for t in items]
            vals = np.array([t[1] for t in items], dtype=np.float32)
            if y not in labs:
                continue
            logits_list.append(vals)
            targets.append(labs.index(y))

    if len(logits_list) == 0:
        # Fallback: leave temperature unchanged
        return float(decision.temperature)

    tau = decision.fit_temperature(logits_list, targets, lr=5e-2, steps=150)
    return float(tau)


def choose_openset_threshold(
    retriever: Retriever,
    decision: DecisionHead,
    known_imgs: List[np.ndarray],
    unknown_imgs: List[np.ndarray],
    k: int = 5,
    target_fpr: float = 0.05,
) -> float:
    """
    Pick an open-set threshold that yields ~target FPR on unknowns.

    Strategy
    - For each image, compute the post-temperature **top-1 probability** using
      the logits approximation described above.
    - Feed positives (correctly recognized knowns) and negatives (unknowns’ top-1)
      into DecisionHead.auto_threshold.
    """
    def collect_scores(imgs: List[np.ndarray], y_labels: Optional[List[str]] = None):
        pos = []
        all_scores = []
        B = 16
        for i in range(0, len(imgs), B):
            rets = retriever.retrieve_batch(imgs[i:i + B])
            for j, ret in enumerate(rets):
                if len(ret.neighbors) == 0:
                    continue
                # build label->score dictionary
                label_scores = {}
                for lbl, sim, _ in ret.neighbors:
                    if not lbl:
                        continue
                    label_scores.setdefault(lbl, []).append(float(sim))
                label_scores = {k: float(np.mean(v)) for k, v in label_scores.items()}
                items = sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True)[:max(2, k)]
                logits = np.array([t[1] for t in items], dtype=np.float32)
                # temperature-scaled softmax
                p = _softmax(logits, temperature=float(decision.temperature))
                top1p = float(p.max())
                all_scores.append(top1p)
                if y_labels is not None:
                    labs = [t[0] for t in items]
                    if y_labels[i + j] in labs:
                        pos.append(top1p)
        return pos, all_scores

    pos_scores, _ = collect_scores(known_imgs, y_labels=None)  # we'll treat all knowns as positives if present in top-k
    _, unk_scores = collect_scores(unknown_imgs, y_labels=None)

    tau = DecisionHead.auto_threshold(pos_scores, unk_scores, target_fpr=target_fpr)
    decision.tau_open = float(tau)
    return float(tau)


# local: light softmax (keep in sync with DecisionHead)
def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64) / max(temperature, 1e-8)
    x -= x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)
