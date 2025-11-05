### ========================================================================================================================================
## Module       : src/eval/bench_rt.py
## Author       : Rizky Johan Saputra (Independent Project)
## Date         : 4th November 2025 (Seoul, South Korea)
## Project      : Vision Fusion Real Time System (Copyright 2025)
## Topics       : Computer Vision, Real-Time Systems, Interactive AI System, NLP, Machine Learning and Memory Augmentation
## Purpose      : Measure, profile, and report the real-time inference performance characteristics
##                of the Vision-Fusion-RT engine (latency distribution, FPS stability, warmup
##                behavior, and overlay rendering cost) using either live webcam or video file input.
## Role         : Real-Time Benchmark Harness
### ========================================================================================================================================

## ======================================================================================================
## SPECIFICATIONS
## ======================================================================================================
"""
Vision-Fusion-RT — Real-Time Benchmark Harness
----------------------------------------------

Goal
- Measure end-to-end latency and throughput (FPS) of the *inference* path using
  a live webcam or a video file. Optionally render overlays and show the stream.

Design
- Uses `io.webcam.FrameGrabber` for live capture OR OpenCV VideoCapture for files.
- Runs a tight loop: grab latest frame -> engine.infer(frame) -> (optional) draw overlay
- Records per-iteration latency and periodic FPS via a smoothed meter.
- Produces a metrics dict with latency summary and per-stage timings if provided.

Non-goals
- We don’t do dataset accuracy here. See `metrics.py` for offline accuracy/CMC/etc.

Requirements
- engine: an instance of `pipeline.inference.InferenceEngine`
- overlay: optional `io.overlay.draw_overlay` function to annotate frames

Example
-------
    from src.eval.bench_rt import run_realtime_benchmark
    stats = run_realtime_benchmark(
        engine=engine,
        source="webcam://0",
        seconds=30,
        target_size=(640,480),
        display=False
    )
    print(stats)
"""

## ======================================================================================================
## SETUP (ADJUSTABLE) (ADJUST IF NECESSARY)
## ======================================================================================================
from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple
from collections import deque
import time
import cv2
import numpy as np

from src.io.webcam import FrameGrabber
from src.io.overlay import draw_overlay
from src.eval.metrics import latency_stats

## ======================================================================================================
## IMPLEMENTATIONS
## ======================================================================================================
# Declare a class to move the average FPS meter in an exponential rate
class FPSMeter:
    """Exponential moving average FPS meter."""
    #
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._last_t = None
        self.fps = 0.0

    #
    def tick(self) -> float:
        #
        now = time.perf_counter()

        #
        if self._last_t is None:
            self._last_t = now
            return self.fps
        
        #
        dt = now - self._last_t
        self._last_t = now
        inst = 1.0 / dt if dt > 1e-6 else 0.0
        self.fps = self.alpha * inst + (1.0 - self.alpha) * self.fps
        return self.fps

#
def _open_source(source: str, size: Tuple[int, int]) -> Tuple[Callable[[], Optional[np.ndarray]], Callable[[], None]]:
    """
    Returns a pair of callables: (read_latest, close)
    - If source starts with "webcam://<id>", use FrameGrabber
    - Else, treat as file path (cv2.VideoCapture)
    """
    #
    if source.startswith("webcam://"):
        #
        cam_id = int(source.split("://")[1])

        #
        grabber = FrameGrabber(device_id=cam_id, size=size, fps=30).start()

        #
        def read_latest():
            fr = grabber.read_latest()
            return None if fr is None else fr.data
        
        #
        def close():
            grabber.stop()

        #
        return read_latest, close

    #
    cap = cv2.VideoCapture(source)

    #
    if size and size[0] > 0 and size[1] > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

    #
    def read_latest():
        ok, frame = cap.read()
        return frame if ok else None

    #
    def close():
        try:
            cap.release()
        except Exception:
            pass
    return read_latest, close

#
def run_realtime_benchmark(
    engine,
    source: str = "webcam://0",
    seconds: int = 20,
    target_size: Tuple[int, int] = (640, 480),
    warmup_s: float = 2.0,
    display: bool = False,
    show_overlay: bool = False,
    overlay_threshold: float = None,
) -> Dict[str, float]:
    """
    Execute an end-to-end benchmark loop for a fixed duration.

    Parameters
    ----------
    engine : InferenceEngine
        Provides .infer(bgr)->InferenceOutput
    source : str
        "webcam://<id>" or a video file path
    seconds : int
        Duration (measurement window), excluding warmup
    target_size : (int,int)
        Requested capture size (best effort)
    warmup_s : float
        Warmup duration before recording stats
    display : bool
        If True, show a window with the stream (Esc to quit)
    show_overlay : bool
        If True, draw label/score/fps overlay
    overlay_threshold : float
        Optional visual indicator for open-set threshold

    Returns
    -------
    dict
        latency summaries + approximate FPS.
    """

    #
    read_latest, close = _open_source(source, target_size)

    #
    t_end_warm = time.perf_counter() + max(0.0, warmup_s)

    #
    while time.perf_counter() < t_end_warm:
        #
        frame = read_latest()

        #
        if frame is None:
            time.sleep(0.01)
            continue
        _ = engine.infer(frame)

    #
    latencies = []
    fpsm = FPSMeter(alpha=0.2)
    t_end = time.perf_counter() + max(1, int(seconds))

    #
    try:
        while time.perf_counter() < t_end:
            frame = read_latest()
            #
            if frame is None:
                time.sleep(0.005)
                continue
        
            #
            t0 = time.perf_counter()
            out = engine.infer(frame)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)
            fps = fpsm.tick()

            #
            if display:
                vis = frame.copy()
                #
                if show_overlay:
                    vis = draw_overlay(
                        vis,
                        label=out.label,
                        score=out.score,
                        fps=fps,
                        show_threshold=overlay_threshold,
                    )
                #
                cv2.imshow("Vision-Fusion-RT (bench)", vis)
                
                #
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    #
    finally:
        close()
        #
        if display:
            #
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    #
    summary = latency_stats(latencies)
    summary["fps_ema"] = fpsm.fps
    summary["duration_s"] = float(seconds)
    summary["n_frames"] = int(len(latencies))
    return summary

### ========================================================================================================================================
## END (ADD IMPLEMENTATIONS IF NECESSARY)
### ========================================================================================================================================