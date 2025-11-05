# src/pipeline/inference.py
"""
Vision-Fusion-RT — Inference Orchestrator
-----------------------------------------

This module glues together:
- Retriever (k-NN with optional text fusion & EMA)
- DecisionHead (temperature + open-set threshold)

Interface
- InferenceEngine(retriever, decision_head)
- .infer(bgr) -> (label, score, dt_ms, debug)
- .set_open_set(enabled: bool)
- .update_text_prototypes(labels: list[str])

`debug` includes neighbors and raw/fused info for UI overlays and logs.

Usage in AppState (pseudo)
    out = engine.infer(frame_bgr)
    overlay = draw_overlay(frame_bgr, out.label, out.score, fps)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np

from src.retrieval.retriever import Retriever
from src.models.heads.decision import DecisionHead
from src.core.types import RetrievalResult


@dataclass
class InferenceOutput:
    label: str
    score: float
    latency_ms: float
    neighbors: List[tuple]           # (label, sim, id)
    extras: Dict[str, Any]


class InferenceEngine:
    def __init__(self, retriever: Retriever, decision: DecisionHead):
        self.retriever = retriever
        self.decision = decision
        # keep an internal switch for open-set (delegated to decision tau)
        self.open_set_enabled = True

    def set_open_set(self, enabled: bool) -> None:
        self.open_set_enabled = bool(enabled)

    def update_text_prototypes(self, labels: List[str]) -> None:
        self.retriever.update_text_prototypes(labels)

    def infer(self, bgr: np.ndarray) -> InferenceOutput:
        t0 = time.perf_counter()

        # 1) Retrieve neighbors + fused label scores (Retriever already does fusion & EMA)
        ret: RetrievalResult = self.retriever.retrieve(bgr)

        # 2) Decision (temperature + open-set)
        # Here we only have the top1 label/score from Retriever.
        # If you want to pass full top-k logits into DecisionHead, add a small
        # extension to Retriever to also return the sorted label list + scores.
        labels = [ret.label]
        scores = np.array([ret.score], dtype=np.float32)

        pred_label, pred_score = self.decision.decide(labels, scores)

        dt_ms = (time.perf_counter() - t0) * 1000.0

        return InferenceOutput(
            label=pred_label if self.open_set_enabled else ret.label,
            score=float(pred_score if self.open_set_enabled else ret.score),
            latency_ms=float(dt_ms),
            neighbors=ret.neighbors,
            extras={
                "raw_top1": (ret.label, ret.score),
                "decision_tau": self.decision.tau_open,
                "temperature": self.decision.temperature,
            },
        )
