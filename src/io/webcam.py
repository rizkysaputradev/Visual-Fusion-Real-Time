# src/io/webcam.py
"""
Vision-Fusion-RT — Webcam I/O (Threaded, Low-Latency)
-----------------------------------------------------

Design goals
- **Threaded capture** with a bounded queue to decouple camera I/O from inference.
- **Frame draining**: always serve the *latest* frame to keep latency flat (no backlog).
- **Auto-reopen** on transient camera failures.
- **Context manager** support (`with FrameGrabber(...) as cam:`).
- **Deterministic size/fps** requests (best effort; depends on driver support).

Notes
- Frames are provided in OpenCV BGR uint8 HxWx3, wrapped in `core.types.Frame`.
- This module intentionally does not depend on torch.

Hot parameters
- `max_queue`: small (e.g., 4) to prevent RAM growth and to enforce "latest frame" semantics.
- `drop_on_backlog`: when the queue is full, pop one before pushing a new frame (keeps freshest).

Example
    from src.io.webcam import FrameGrabber
    with FrameGrabber(device_id=0, size=(640,480), fps=30).start() as cam:
        while True:
            fr = cam.read_latest()
            if fr is None: continue
            # use fr.data (BGR), fr.ts, fr.size
"""

from __future__ import annotations
import cv2
import threading
import queue
import time
import sys
from typing import Optional, Tuple

import numpy as np
from src.core.types import Frame


class FrameGrabber:
    """
    Threaded webcam capture with bounded queue and "latest frame" reads.

    Parameters
    ----------
    device_id : int
        OpenCV camera index (0 for default).
    size : (int, int)
        Desired frame size (width, height). Driver may return nearest match.
    fps : int
        Desired frames per second (best effort).
    max_queue : int
        Queue capacity; small to bound latency and memory.
    drop_on_backlog : bool
        If True, when queue is full, pop one item before pushing a new frame (keep newest).
    reopen_interval_s : float
        If capture fails, attempt reopen no more frequently than this.

    Thread-safety
    -------------
    - Public methods (`start`, `read_latest`, `stop`) are safe to call from the main thread.
    - The capture thread is daemonized and cleans up on `stop()` or object GC.
    """

    def __init__(
        self,
        device_id: int = 0,
        size: Tuple[int, int] = (640, 480),
        fps: int = 30,
        max_queue: int = 4,
        drop_on_backlog: bool = True,
        reopen_interval_s: float = 1.0,
    ):
        self.device_id = int(device_id)
        self.req_size = (int(size[0]), int(size[1]))
        self.req_fps = int(fps)
        self.max_queue = int(max_queue)
        self.drop_on_backlog = bool(drop_on_backlog)
        self.reopen_interval_s = float(reopen_interval_s)

        self._q: "queue.Queue[Frame]" = queue.Queue(maxsize=self.max_queue)
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_open_attempt = 0.0

    # ---- Lifecycle ---------------------------------------------------------

    def start(self) -> "FrameGrabber":
        """Open device and start capture thread."""
        self._open()
        if not self._th.is_alive():
            self._th.start()
        return self

    def stop(self) -> None:
        """Stop thread and release camera."""
        self._stop.set()
        if self._th.is_alive():
            self._th.join(timeout=0.5)
        self._close()

    # context manager support
    def __enter__(self) -> "FrameGrabber":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    # ---- Public API --------------------------------------------------------

    def read_latest(self) -> Optional[Frame]:
        """
        Non-blocking read of the **latest** frame (drains stale frames).
        Returns None if queue is empty.
        """
        try:
            # Drain backlog to keep latency minimal.
            while self._q.qsize() > 1:
                self._q.get_nowait()
            return self._q.get_nowait()
        except queue.Empty:
            return None

    # ---- Internals ---------------------------------------------------------

    def _open(self) -> None:
        """Open and configure the camera (best-effort)."""
        # Pick a stable backend per-OS
        backend = cv2.CAP_ANY
        if sys.platform == "darwin":
            backend = cv2.CAP_AVFOUNDATION   # crucial for macOS stability
        # (Windows users would prefer CAP_DSHOW; Linux often fine with CAP_V4L2/CAP_ANY)

        self._cap = cv2.VideoCapture(self.device_id, backend)
        if not self._cap or not self._cap.isOpened():
            if self._cap:
                self._cap.release()
            self._cap = None
            return

        # Request properties (drivers may clamp to supported values)
        w, h = self.req_size
        # set a few times; some cams ignore the first call
        for _ in range(3):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(w))
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
            self._cap.set(cv2.CAP_PROP_FPS,          float(self.req_fps))
            ok, _ = self._cap.read()
            if ok:
                break

        # Warm-up: drain a few frames so exposure/white-balance settle
        # (and to verify the stream is really delivering frames)
        for _ in range(8):
            ok, _ = self._cap.read()
            if not ok:
                break

    def _close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def _maybe_reopen(self) -> None:
        now = time.perf_counter()
        if (now - self._last_open_attempt) < self.reopen_interval_s:
            return
        self._last_open_attempt = now
        self._close()
        self._open()

    def _loop(self) -> None:
        """Capture loop: read frames, enqueue; on failure, try to reopen periodically."""
        while not self._stop.is_set():
            if self._cap is None:
                # try reopen with backoff
                self._maybe_reopen()
                time.sleep(0.05)
                continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                # camera might have been disconnected or hiccuped
                time.sleep(0.05)
                self._maybe_reopen()
                continue

            ts = time.perf_counter()
            h, w = frame.shape[:2]
            fr = Frame(data=frame, ts=ts, size=(w, h))

            if self.drop_on_backlog and self._q.full():
                try:
                    self._q.get_nowait()  # drop oldest
                except queue.Empty:
                    pass
            try:
                self._q.put_nowait(fr)
            except queue.Full:
                # rare: if another consumer raced, just skip this frame
                pass
