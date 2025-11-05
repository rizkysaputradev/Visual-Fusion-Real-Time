### ========================================================================================================================================
## Module       : src/core/logging_utils.py
## Author       : Rizky Johan Saputra (Independent Project)
## Date         : 4th November 2025 (Seoul, South Korea)
## Project      : Vision Fusion Real Time System (Copyright 2025)
## Topics       : Computer Vision, Real-Time Systems, Interactive AI System, NLP, Machine Learning and Memory Augmentation
## Purpose      : Centralize lightweight logging and performance utilities: a configured logger,
##                a context timer for micro-benchmarks, an EMA-based FPS meter, runtime log-level
##                switching, and a small accumulator for step-wise timing summaries.
## Role         : Logging & Timing Utilities
### ========================================================================================================================================

## ======================================================================================================
## SPECIFICATIONS
## ======================================================================================================
"""
Vision-Fusion-RT — Logging & Timing Utilities
---------------------------------------------

- Centralized `logger` configured on import (INFO by default).
- `timer(name)` context manager for micro-benchmarks (ms).
- `FPSMeter` — exponential moving average FPS tracker.
- `setup_logging(level)` — change level at runtime.
- `PerfAccumulator` — accumulate and summarize timing stats across steps.
"""

## ======================================================================================================
## SETUP (ADJUSTABLE) (ADJUST IF NECESSARY)
## ======================================================================================================
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional
import logging
import sys
import time

## ======================================================================================================
## IMPLEMENTATIONS
## ======================================================================================================
# Configure a root log format and create a project-scoped logger
_LOG_FORMAT = "[%(levelname)s] %(asctime)s %(name)s:%(lineno)d - %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger("vf-rt")

# Setup the logger and adjust in a global level
def setup_logging(level: str = "INFO") -> None:
    # Map the string level to logging constant (fallback to INFO)
    level_val = getattr(logging, level.upper(), logging.INFO)

    # Update all the root handlers so third-party libraries accepts the new level
    for lh in logging.getLogger().handlers:
        lh.setLevel(level_val)
    
    # Update the root and module logger
    logging.getLogger().setLevel(level_val)
    logger.setLevel(level_val)

# Declare a simple context timer that logs the duration at a debug level
@contextmanager
def timer(name: str):
    # Capture the start time at high resolution
    t0 = time.perf_counter()
    try:
        yield
    finally:
        # Compute the elapsed milliseconds and log at a debug level
        dt = (time.perf_counter() - t0) * 1000.0
        logger.debug(f"[timer] {name}: {dt:.2f} ms")

# Assign a class to compute the exponential based moving-average FPS meter
class FPSMeter:
    # Initialize the beta (Higher will result in a smoother but slower reactivity)
    def __init__(self, beta: float = 0.9):
        # Smoothing factor and internal state
        self.beta = float(beta)
        self._fps = 0.0
        self._last: Optional[float] = None

    # Declare an advancement on a single frame and smoothen the FPS estimation
    def step(self) -> float:
        # Assign the current timestamp
        now = time.perf_counter()

        # Initial call to bootstrap the clock
        if self._last is None:
            self._last = now
            return self._fps
        
        # Compute the instantaneous FPS and smoothened FPS with EMA-update
        dt = max(now - self._last, 1e-6)
        self._last = now
        inst = 1.0 / dt
        self._fps = self.beta * self._fps + (1 - self.beta) * inst
        return self._fps

# Assign a dataclass that aggregates the timing samples and provides rolling statistics
@dataclass
class PerfAccumulator:
    # Assign the dicitonary for the counter and total durations by name
    counts: Dict[str, int] = field(default_factory=dict)
    sums_ms: Dict[str, float] = field(default_factory=dict)

    # Assign the dicitonary for the track per-name extrema for quick diagnostics
    max_ms: Dict[str, float] = field(default_factory=dict)
    min_ms: Dict[str, float] = field(default_factory=dict)

    # Measure the block under a given name and record stats with context manager
    @contextmanager
    def measure(self, name: str):
        # Ensure to start the timestamp
        t0 = time.perf_counter()
        try:
            yield
        finally:

            # Elapsed the ms and running aggregates
            dt = (time.perf_counter() - t0) * 1000.0
            self.counts[name] = self.counts.get(name, 0) + 1
            self.sums_ms[name] = self.sums_ms.get(name, 0.0) + dt
            self.max_ms[name] = max(self.max_ms.get(name, dt), dt)
            self.min_ms[name] = min(self.min_ms.get(name, dt), dt) if name in self.min_ms else dt

    # Declare an interpretable summary
    def summary(self, reset: bool = False) -> str:
        # Construct a multi-line report with the metrics per key (Avg, min, max, sum)
        lines = ["Perf Summary (ms):"]
        for k in sorted(self.counts.keys()):
            n = self.counts[k]
            s = self.sums_ms[k]
            avg = s / max(n, 1)
            lo = self.min_ms[k]
            hi = self.max_ms[k]
            lines.append(f"  - {k}: n={n} avg={avg:.2f} min={lo:.2f} max={hi:.2f} sum={s:.2f}")
        out = "\n".join(lines)

        # Define the reset to start a fresh measurement window
        if reset:
            self.counts.clear(); self.sums_ms.clear(); self.max_ms.clear(); self.min_ms.clear()
        return out

### ========================================================================================================================================
## END (ADD IMPLEMENTATIONS IF NECESSARY)
### ========================================================================================================================================