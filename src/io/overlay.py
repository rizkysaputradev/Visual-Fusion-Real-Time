# src/io/overlay.py
"""
Vision-Fusion-RT — Visualization Overlay
----------------------------------------

Responsibilities
- Draw a clean, legible status bar with label, score, and FPS.
- Optional bounding boxes / text annotations for future detectors.
- Anti-flicker: slight background panel to maintain readability.

Design
- Uses neutral palette (paper-like background, dark text).
- Works on any BGR ndarray (OpenCV).
"""

from __future__ import annotations
from typing import Iterable, Optional, Tuple
import cv2
import numpy as np


# Neutral palette
_BG = (245, 245, 240)   # panel background
_FG = (20, 20, 20)      # text color
_ACC = (60, 60, 60)     # accents / borders


def draw_text_bar(
    frame: np.ndarray,
    left_text: str,
    right_text: Optional[str] = None,
    top: int = 10,
    height: int = 40,
    pad: int = 12,
) -> np.ndarray:
    """
    Draw a text bar along the top of the frame.
    """
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (10, top), (w - 10, top + height), _BG, thickness=-1)
    cv2.putText(frame, left_text, (10 + pad, top + int(0.7 * height)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, _FG, 2, cv2.LINE_AA)
    if right_text:
        (tw, th), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(frame, right_text, (w - 10 - pad - tw, top + int(0.7 * height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, _FG, 2, cv2.LINE_AA)
    return frame


def draw_overlay(
    frame: np.ndarray,
    label: str,
    score: float,
    fps: float,
    show_threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Compact status overlay: "<label> (score) • <fps> FPS".
    """
    left = f"{label} ({score:.2f})"
    right = f"{fps:.1f} FPS"
    out = draw_text_bar(frame, left_text=left, right_text=right)
    if show_threshold is not None:
        # thin line indicating open-set threshold scale
        w = frame.shape[1]
        x0, x1 = int(0.1 * w), int(0.9 * w)
        y = 60
        cv2.line(out, (x0, y), (x1, y), _ACC, 2)
        xt = x0 + int((x1 - x0) * max(0.0, min(1.0, show_threshold)))
        cv2.circle(out, (xt, y), 4, _FG, -1)
    return out


def draw_boxes(
    frame: np.ndarray,
    boxes: Iterable[Tuple[int, int, int, int]],
    labels: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """
    Draw one or more bounding boxes on the frame. Boxes are (x1, y1, x2, y2).
    If labels provided, overlay each near the top-left corner of the box.
    """
    if labels is None:
        labels = [""] * len(list(boxes))
    for (x1, y1, x2, y2), text in zip(boxes, labels):
        cv2.rectangle(frame, (x1, y1), (x2, y2), _FG, 2)
        if text:
            cv2.putText(frame, text, (x1 + 4, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _FG, 2, cv2.LINE_AA)
    return frame
