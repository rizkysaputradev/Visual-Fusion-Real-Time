# src/models/heads/decision.py
"""
Vision-Fusion-RT — Decision Head (Calibration + Open-Set)
---------------------------------------------------------

Input
- Ranked labels and their similarity scores (cosine/IP). These typically come
  from the retriever after fusing image and (optional) text prototype scores.

Output
- Final (label, score) decision with:
  * Temperature scaling (softmax temperature τ > 0; lower τ → sharper)
  * Open-set rejection based on calibrated confidence threshold
  * Optional margin-based unknown (top1 - top2 < δ)

API
- DecisionHead(tau_open=0.28, temperature=0.9, use_margin=False, margin_delta=0.05)
- decide(labels, scores) -> (label, score)
- fit_temperature(logit_list, target_list) -> optimize τ on a dev set (NLL)
- auto_threshold(pos_scores, neg_scores, target_fpr=0.05) -> pick τ_open

Notes
- We assume `scores` are *similarities* (higher is better). To derive a confidence,
  we apply temperature-scaled softmax over the top-k scores.
- If you prefer energy-based open set (E = -logsumexp), you can extend this class easily.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import math
import numpy as np


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64) / max(temperature, 1e-8)
    x = x - np.max(x)  # stability
    e = np.exp(x)
    p = e / np.sum(e)
    return p.astype(np.float32)


@dataclass
class DecisionHead:
    tau_open: float = 0.28           # open-set threshold on calibrated P(top1)
    temperature: float = 0.9         # temperature for softmax over similarities
    use_margin: bool = False         # enable margin-based unknown
    margin_delta: float = 0.05       # margin threshold (top1 - top2 < δ ⇒ unknown)

    def decide(self, labels: Sequence[str], scores: np.ndarray) -> Tuple[str, float]:
        """
        Compute final decision from candidate labels and similarity scores.

        Returns
        -------
        (label, score)
            label: predicted class or "unknown"
            score: calibrated top-1 confidence (softmax prob)
        """
        labels = list(labels)
        if len(labels) == 0 or len(scores) == 0:
            return "unknown", 0.0

        scores = np.asarray(scores, dtype=np.float32)
        # Softmax probabilities with temperature
        probs = _softmax(scores, temperature=float(self.temperature))
        idx = int(np.argmax(probs))
        top1_label, top1_p = labels[idx], float(probs[idx])

        # Optional margin unknown
        if self.use_margin and len(probs) >= 2:
            top2 = float(np.partition(probs, -2)[-2])
            if (top1_p - top2) < float(self.margin_delta):
                return "unknown", top1_p

        # Open-set rejection
        if top1_p < float(self.tau_open):
            return "unknown", top1_p

        return top1_label, top1_p

    # ---------------------------------------------------------------------
    # Calibration utilities (dev-set driven; optional)
    # ---------------------------------------------------------------------

    def fit_temperature(self, logit_list: List[np.ndarray], target_list: List[int],
                        lr: float = 1e-1, steps: int = 200) -> float:
        """
        Fit temperature by minimizing NLL on a small dev set.

        Parameters
        ----------
        logit_list : list of arrays
            Each array is [k] similarities for a sample (order must align with targets)
        target_list : list[int]
            Index of the correct class within each corresponding logit array.
        lr : float
            Learning rate for simple gradient descent on τ.
        steps : int
            Number of iterations.

        Returns
        -------
        float
            Fitted temperature.
        """
        # Initialize with current temperature
        tau = max(1e-3, float(self.temperature))

        def nll_and_grad(t: float) -> Tuple[float, float]:
            nll = 0.0
            grad = 0.0
            for logits, y in zip(logit_list, target_list):
                logits = np.asarray(logits, dtype=np.float64)
                z = logits / max(t, 1e-8)
                z = z - np.max(z)
                e = np.exp(z)
                p = e / np.sum(e)
                nll -= math.log(max(p[y], 1e-12))
                # gradient d/dt of -log p_y with respect to t
                # d/dt softmax(z/t) = (softmax * (sum_j p_j * z_j - z_y)) / t
                ez = np.sum(p * z)
                grad += (ez - z[y]) / max(t, 1e-8)
            n = max(1, len(logit_list))
            return nll / n, grad / n

        for _ in range(max(1, steps)):
            nll, g = nll_and_grad(tau)
            tau -= lr * g
            tau = float(max(1e-3, min(10.0, tau)))  # clamp to reasonable range

        self.temperature = tau
        return tau

    @staticmethod
    def auto_threshold(pos_scores: Iterable[float], neg_scores: Iterable[float],
                       target_fpr: float = 0.05) -> float:
        """
        Choose a threshold τ* on confidence scores that yields approximately
        target FPR on negatives (unknowns).

        Parameters
        ----------
        pos_scores : iterable of float
            Scores (probabilities) for *correctly* recognized knowns.
        neg_scores : iterable of float
            Scores predicted for *unknowns* (should be low ideally).
        target_fpr : float
            Desired false-positive rate for unknowns.

        Returns
        -------
        float
            Chosen threshold τ_open.
        """
        pos = np.sort(np.asarray(list(pos_scores), dtype=np.float64))
        neg = np.sort(np.asarray(list(neg_scores), dtype=np.float64))

        if neg.size == 0:
            # No unknowns provided; fallback to conservative percentile on positives
            return float(np.percentile(pos, 10)) if pos.size > 0 else 0.5

        # For a threshold tau, FPR = P(neg >= tau).
        # So pick tau as the (1 - target_fpr) percentile of negatives.
        q = max(0.0, min(1.0, 1.0 - float(target_fpr)))
        tau = float(np.quantile(neg, q))
        return tau
