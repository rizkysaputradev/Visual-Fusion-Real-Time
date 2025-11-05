### ========================================================================================================================================
## Module       : src/app/main_rt.py
## Author       : Rizky Johan Saputra (Independent Project)
## Date         : 4th November 2025 (Seoul, South Korea)
## Project      : Vision Fusion Real Time System (Copyright 2025)
## Topics       : Computer Vision, Real-Time Systems, Interactive AI System, NLP, Machine Learning and Memory Augmentation
## Purpose      : Orchestrates the end-to-end real-time loop: configure devices/models, grab webcam frames,
##                embed and retrieve with image/text fusion, apply open-set decisions, render overlays, and
##                support interactive few-shot registration, snapshots, and text-prototype updates via hotkeys.
## Role         : Real-Time Orchestrator
### ========================================================================================================================================

## ======================================================================================================
## SPECIFICATIONS
## ======================================================================================================
"""
Vision-Fusion-RT — Real-Time Orchestrator
-----------------------------------------
Responsibilities
- Initialize config, logging, device, models, memory, retriever, decision head.
- Launch a threaded webcam capture with a bounded queue (frame-draining to keep latency flat).
- Run an inference loop at real-time speeds (target ~30 FPS) with:
    - Image → embedding → retrieval → (optional) text fusion → decision
    - EMA-based temporal smoothing for stability (optional)
    - Open-set thresholding and “unknown” handling
- Provide keyboard hotkeys for:
    [q] Quit                           
    [s] Save snapshot (PNG)
    [r] Register class from buffered frames
    [u] Undo last registration for the most recent class
    [o] Toggle open-set rejection on/off
    [t] Update/Add text prototype(s) via console prompt
- Maintain a ring buffer of recent frames for few-shot registration
- Clean teardown on exit

Note
- This module references the following layers you will implement next:
    src/core/{config,logging_utils,types,utils}.py
    src/io/{webcam,image_io,overlay}.py
    src/models/backbones/registry.py
    src/models/text/registry.py
    src/models/heads/decision.py
    src/memory/incremental.py      (store_router.py may later replace direct construction)
    src/retrieval/{retriever.py,fusion.py}

CLI
    python -m src.app.main_rt --config experiments/configs/default.yaml
    python -m src.app.main_rt --device mps --cam 0 --size 640 480 --fps 30

Design choices
- We drain the capture queue before reading to avoid stale frames causing latency growth.
- Registration uses the most recent N frames (light data augmentation will be handled in pipeline/register.py).
- We cache text embeddings per label string to avoid re-encoding every frame.
"""

## ======================================================================================================
## SETUP (ADJUSTABLE) (ADJUST IF NECESSARY)
## ======================================================================================================
from __future__ import annotations
import argparse
import collections
import os
import sys
import time
import json
from typing import Dict, List, Tuple, Optional
from src.retrieval.retriever import Retriever, RetrieverConfig

# Webcam
import cv2
import numpy as np
from PIL import Image

# Core & IO
from src.core.config import load_config, AppConf
from src.core.logging_utils import logger, FPSMeter, timer
from src.core.utils import get_device, seed_everything, l2_normalize
from src.core.types import Frame
from src.io.webcam import FrameGrabber
from src.io.overlay import draw_overlay
from src.io.image_io import to_pil, imwrite

# Models (registries will be implemented in the next module)
try:
    from src.models.backbones.registry import build_backbone
    from src.models.text.registry import build_text_encoder
except Exception:
    build_backbone = None
    build_text_encoder = None

# Memory & Retrieval
from src.memory.incremental import IncrementalMemory
from src.models.heads.decision import DecisionHead

# Retriever should support fusion
from src.retrieval.retriever import Retriever

## ======================================================================================================
## IMPLEMENTATIONS
## ======================================================================================================
# Convert OpenCV BGR ndarray to PIL RGB image
def _ndarray_to_pil_bgr(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(frame_bgr[:, :, ::-1])

# Convert the string to its local time
def _now_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

# Caches text embeddings keyed by label string
class TextProtoCache:
    """
    Caches text embeddings keyed by label string.
    Uses a provided text encoder with an .encode_text(list[str]) -> np.ndarray interface.
    All vectors are L2-normalized.
    """

    # Create a small in-memory cache for text and embedding lookups
    def __init__(self, text_encoder):
        self.enc = text_encoder
        self.cache: Dict[str, np.ndarray] = {}

    # Retrieve the prototype vector for one label
    def get(self, label: str) -> np.ndarray:
        if label not in self.cache:
            vec = self.enc.encode_text([label])[0]
            self.cache[label] = vec.astype("float32")
        return self.cache[label]
    
    # Batch-get multiple label vectors and encoding only missing vectors
    def get_many(self, labels: List[str]) -> Dict[str, np.ndarray]:
        to_encode = [l for l in labels if l not in self.cache]
        if to_encode:
            arr = self.enc.encode_text(to_encode).astype("float32")
            for i, l in enumerate(to_encode):
                self.cache[l] = arr[i]
        return {l: self.cache[l] for l in labels}
    
    # Manually set or override a label’s prototype vector
    def set(self, label: str, vec: np.ndarray) -> None:
        self.cache[label] = vec.astype("float32")

    # Remove a label from the cache (forces re-encode on next access)
    def remove(self, label: str) -> None:
        if label in self.cache:
            del self.cache[label]

# Holds long-lived objects and mutable session state for the RT loop
class AppState:
    """
    Holds long-lived objects and mutable session state for the RT loop.
    """
    # Initialize the configs for the app
    def __init__(self, cfg: AppConf):
        self.cfg = cfg
        self.device = str(get_device(cfg.device))
        seed_everything(cfg.seed)

        # Build the encoders with an error fallback
        if build_backbone is None or build_text_encoder is None:
            raise ImportError(
                "Model registries not found. Implement models/backbones/registry.py and models/text/registry.py next."
            )
        
        # Instantiate the image and text encoders (Require backbone to expose embedding dimension)
        self.vision = build_backbone(cfg.backbone, device=self.device)
        self.textenc = build_text_encoder(cfg.text_encoder, device=self.device)
        self.embed_dim = getattr(self.vision, "dim", None)
        if self.embed_dim is None:
            raise RuntimeError("Backbone must expose attribute `.dim` for embedding size.")

        # Dynamic memory calling (Memory increments are directly accessed for simplification)
        self.memory = IncrementalMemory(dim=self.embed_dim, metric=cfg.memory.metric)

        # Retriever and Decision
        r_cfg = RetrieverConfig(
            k=cfg.retrieval.k,
            alpha_fusion=cfg.retrieval.alpha_fusion,
            temporal_ema=cfg.retrieval.temporal_ema,
            neighbor_agg=cfg.retrieval.neighbor_agg,
            text_prompt_template="{label}",
        )

        # Assembled retriever with image+text encoders and bound memory
        self.retriever = Retriever(
            memory=self.memory,
            image_encoder=self.vision,
            text_encoder=self.textenc,
            text_aligned=True,
            cfg=r_cfg,
        )

        # Retain the instantiated decision head and an open set threshold
        self.decision = DecisionHead(
            tau_open=cfg.decision.open_set_threshold,
            temperature=cfg.decision.temperature
        )

        # Prototypes (text) cache
        self.txt_cache = TextProtoCache(self.textenc)

        # Label set and last predicted state for temporal EMA smoothing
        self.known_labels: List[str] = []
        self.last_score_map: Dict[str, float] = {}

        # Buffer the recent frames for registration (Few-shot)
        self.buffer = collections.deque(maxlen=64)

        # FPS meter for display
        self.fps_meter = FPSMeter(beta=0.9)

        # Open-set toggle
        self.open_set_enabled = True

        # Snapshots direectory
        self.snap_dir = os.path.join("experiments", "results", "snapshots")
        os.makedirs(self.snap_dir, exist_ok=True)

    # Ensure every label is tracked and retriever has up-to-date text prototypes
    def ensure_labels(self, labels: List[str]) -> None:
        for l in labels:
            if l not in self.known_labels:
                self.known_labels.append(l)

        # Let the retriever re-encode the text prompts for its labels
        self.retriever.update_text_prototypes(self.known_labels)

    # Register the latest buffered frames as samples for a new/existing class label
    def register_from_buffer(self, label: str, num_frames: int = 6, stride: int = 2) -> int:
        # Ensure that there are available frames to register
        if not self.buffer:
            logger.warning("Registration buffer is empty; show the object to the camera first.")
            return 0

        # Retrieve the last N frames with stride (Sample frames)
        idxs = list(range(len(self.buffer) - 1, -1, -stride))[:num_frames]
        frames = [self.buffer[i] for i in idxs][::-1]

        # Convert to PIL and encode with a L2-regularization
        imgs = [_ndarray_to_pil_bgr(fr.data) for fr in frames]
        with timer("encode_register_images"):
            vecs = self.vision.encode_images(imgs)
            vecs = vecs.astype("float32")

        # Attach lightweight metadata and store in memory
        metas = [{"source": "rt", "ts": fr.ts} for fr in frames]
        added_ids = self.memory.register_class(label, vecs, metas)

        # Ensure that the label is known and prototypes are refreshed
        self.ensure_labels([label])
        logger.info(f"Registered label='{label}' with {len(added_ids)} vectors (mem.ntotal={self.memory.store.ntotal}).")
        return len(added_ids)

    # Placeholder for undo and implemented later alongside memory/store_router and a label→ids map
    def undo_last_for(self, label: str) -> None:
        logger.warning("Undo is not yet implemented for Faiss Flat store. (Will be enabled after id mapping & remove())")

    # Force refresh or creation of text prototypes for provided labels
    def update_text_prototypes(self, labels: List[str]) -> None:
        self.ensure_labels(labels)
        _ = self.txt_cache.get_many(labels)  # encode & cache
        logger.info(f"Updated text prototypes for: {labels}")

    # Run a single real-time inference step on a BGR frame (returns label, score, latency_ms)
    def infer(self, frame_bgr: np.ndarray) -> Tuple[str, float, float]:
        # Initialize the timer counter
        t0 = time.perf_counter()

        # Ensure that the text prototypes exist for the current labels
        if self.known_labels:
            self.retriever.update_text_prototypes(self.known_labels)

        # returns the label, score, neighbors and latency
        ret = self.retriever.retrieve(frame_bgr)

        # Compute and assign the label and scores
        label = ret.label
        score = float(ret.score)

        # Define the open-set by flipping to unknown if it is enabled and score below the threshold 
        if self.open_set_enabled and score < float(self.cfg.decision.open_set_threshold):
            label, score = "unknown", 0.0

        # Measure the end-to-end latency for diagnostics
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return label, score, dt_ms
    
# Declare the function to run the system or simulation with the set parameterizations
def run(cfg: AppConf, args) -> None:
    state = AppState(cfg)
    logger.info(f"Device={state.device}, backbone={cfg.backbone}, text_encoder={cfg.text_encoder}, dim={state.embed_dim}")

    # Preload demo labels unless its provided later (Adjustable for research)
    if args.init_labels:
        init_labels = [l.strip() for l in args.init_labels.split(",") if l.strip()]
        state.ensure_labels(init_labels)
        logger.info(f"Initial labels: {init_labels}")
    else:
        demo = ["bottle", "cup", "phone"]
        state.ensure_labels(demo)
        logger.info(f"Demo labels: {demo}")

    # Start the camera
    grab = FrameGrabber(device_id=cfg.rt.cam_device,
                        size=cfg.rt.cam_size,
                        fps=cfg.rt.cam_fps,
                        max_queue=cfg.rt.max_queue,
                        drop_on_backlog=cfg.rt.drop_frame_on_backlog).start()
    logger.info(f"Camera started: device={cfg.rt.cam_device} size={cfg.rt.cam_size} fps={cfg.rt.cam_fps}")

    # Declare the last time and FPS (Adjustable, assign the last time when memory router is implemented)
    last_ts = time.perf_counter()
    fps = 0.0
    try:
        while True:
            fr: Optional[Frame] = grab.read_latest()
            if fr is None:
                time.sleep(0.002)
                continue

            # Buffer frame for potential registration
            state.buffer.append(fr)

            # Inference
            label, score, dt = state.infer(fr.data)

            # FPS
            fps = state.fps_meter.step()

            # Overlay & display
            vis = draw_overlay(fr.data.copy(), label=label, score=score, fps=fps)
            cv2.imshow("Vision-Fusion-RT — press [q] to quit, [r] register, [s] snapshot, [o] open-set toggle, [t] text proto", vis)
            key = cv2.waitKey(1) & 0xFF

            # Assign the keyboard inputs for the interactive features
            if key == ord('q'):
                logger.info("Quit signal.")
                break
            elif key == ord('s'):
                # Save the snapshot
                fn = os.path.join(state.snap_dir, f"{_now_str()}_{label}.png")
                imwrite(fn, fr.data)
                logger.info(f"Saved snapshot: {fn}")
            elif key == ord('r'):
                # Register from its buffers
                new_label = args.reg_label or input("Enter new class label to register: ").strip()
                if not new_label:
                    logger.warning("Empty label. Cancelled.")
                else:
                    n = state.register_from_buffer(new_label, num_frames=args.reg_frames, stride=args.reg_stride)
                    logger.info(f"Registered {n} samples under label '{new_label}'.")
            elif key == ord('u'):
                # Undo last (Requires id book-keeping in memory store)
                tgt = args.reg_label or input("Enter label to undo last from: ").strip()
                if tgt:
                    state.undo_last_for(tgt)
            elif key == ord('o'):
                state.open_set_enabled = not state.open_set_enabled
                logger.info(f"Open-set rejection {'ENABLED' if state.open_set_enabled else 'DISABLED'}.")
            elif key == ord('t'):
                # Update or add text prototypes interactively
                raw = input("Enter comma-separated labels to (re)encode text prototypes: ").strip()
                labels = [s.strip() for s in raw.split(",") if s.strip()]
                if labels:
                    state.update_text_prototypes(labels)
                else:
                    logger.info("No labels provided.")
    
    # Declare a keyboard input interruption to the system
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        grab.stop()
        cv2.destroyAllWindows()
        logger.info("Camera released; windows closed.")

# Declare the CLI and Entry Point Parser
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Vision-Fusion-RT — Real-Time App")
    p.add_argument("--config", type=str, default="experiments/configs/default.yaml", help="Path to YAML config.")
    p.add_argument("--device", type=str, default=None, help="Override device: cuda|mps|cpu")
    p.add_argument("--seed", type=int, default=None, help="Override RNG seed.")
    p.add_argument("--cam", type=int, default=None, help="Override camera device id.")
    p.add_argument("--size", type=int, nargs=2, default=None, help="Override camera size W H.")
    p.add_argument("--fps", type=int, default=None, help="Override camera FPS.")
    p.add_argument("--init-labels", type=str, default="", help="Comma-separated label names to preload text prototypes.")
    p.add_argument("--reg-label", type=str, default="", help="Default label name when pressing [r]; if empty, prompt.")
    p.add_argument("--reg-frames", type=int, default=6, help="Number of frames to use per registration.")
    p.add_argument("--reg-stride", type=int, default=2, help="Stride through buffer for registration sampling.")
    return p

# Declare a simple CLI override system into the app configs
def apply_overrides(cfg: AppConf, args) -> AppConf:
    if args.device:  cfg.device = args.device
    if args.seed is not None: cfg.seed = int(args.seed)
    if args.cam is not None:  cfg.rt.cam_device = int(args.cam)
    if args.size is not None: cfg.rt.cam_size = (int(args.size[0]), int(args.size[1]))
    if args.fps is not None:  cfg.rt.cam_fps = int(args.fps)
    return cfg

# Declare the main system runner
def main():
    ap = build_argparser()
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    run(cfg, args)

## ======================================================================================================
## DEMO AND TESTING
## ======================================================================================================
if __name__ == "__main__":
    main()

### ========================================================================================================================================
## END (ADD IMPLEMENTATIONS IF NECESSARY)
### ========================================================================================================================================