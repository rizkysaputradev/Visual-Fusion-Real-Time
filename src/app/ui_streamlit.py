### ========================================================================================================================================
## Module       : src/app/ui_streamlit.py
## Author       : Rizky Johan Saputra (Independent Project)
## Date         : 4th November 2025 (Seoul, South Korea)
## Project      : Vision Fusion Real Time System (Copyright 2025)
## Topics       : Computer Vision, Real-Time Systems, Interactive AI System, NLP, Machine Learning and Memory Augmentation
## Purpose      : Provide an interactive Streamlit dashboard for Vision-Fusion-RT to start/stop the live pipeline,
##                visualize overlays (label/score/FPS), manage labels and few-shot registration from the buffer,
##                tune retrieval/decision hyperparameters (k, α-fusion, EMA, temperature, open-set τ), and save snapshots,
##                all while reusing the shared AppState for consistent orchestration.
## Role         : Streamlit UI
### ========================================================================================================================================

## ======================================================================================================
## SPECIFICATIONS
## ======================================================================================================
"""
Vision-Fusion-RT — Streamlit UI
--------------------------------
Interactive dashboard around the real-time pipeline.

Features
- Start/Stop camera and show live inference (overlay with label, score, FPS)
- Manage labels: preload, add, re-encode text prototypes
- Few-shot registration from recent frame buffer (N frames, stride)
- Toggle open-set rejection, tune k, alpha fusion, temperature, threshold
- Save snapshots
- Uses the same components as main_rt.py (AppState, FrameGrabber, etc.)

Run
    streamlit run -m src.app.ui_streamlit
or  python -m src.app.ui_streamlit  (Streamlit will still pick it up if installed)
"""

## ======================================================================================================
## SETUP (ADJUSTABLE) (ADJUST IF NECESSARY)
## ======================================================================================================
from __future__ import annotations
import os, time
import numpy as np
from typing import Dict, List, Optional

import streamlit as st
import cv2
from PIL import Image

# Core / IO / Models / Memory / Retrieval
from src.core.config import load_config, AppConf
from src.core.logging_utils import logger, FPSMeter
from src.core.utils import get_device
from src.io.webcam import FrameGrabber
from src.io.overlay import draw_overlay
from src.io.image_io import imwrite

# Orchestrator state (Reuse the AppState from main_rt.py for consistency)
from src.app.main_rt import AppState

## ======================================================================================================
## IMPLEMENTATIONS
## ======================================================================================================
# Declare the session-scoped helpers and page composition
def _ensure_session():
    # Initialize persistent objects in Streamlit session_state
    if "cfg" not in st.session_state:
        cfg_path = os.environ.get("VFRT_CONFIG", "experiments/configs/default.yaml")
        st.session_state.cfg = load_config(cfg_path)
    if "state" not in st.session_state:
        st.session_state.state: Optional[AppState] = None
    if "grab" not in st.session_state:
        st.session_state.grab: Optional[FrameGrabber] = None
    if "running" not in st.session_state:
        st.session_state.running: bool = False
    if "fps" not in st.session_state:
        st.session_state.fps = FPSMeter(beta=0.9)

# Spin up the AppState and camera grabber (Mark the pipeline as running)
def _start_pipeline():
    cfg: AppConf = st.session_state.cfg
    if st.session_state.state is None:
        st.session_state.state = AppState(cfg)
    if st.session_state.grab is None:
        st.session_state.grab = FrameGrabber(
            device_id=cfg.rt.cam_device,
            size=cfg.rt.cam_size,
            fps=cfg.rt.cam_fps,
            max_queue=cfg.rt.max_queue,
            drop_on_backlog=cfg.rt.drop_frame_on_backlog
        ).start()
    st.session_state.running = True
    logger.info("[UI] Streamlit: pipeline started.")

# Stop the streaming process and release its resources
def _stop_pipeline():
    if st.session_state.grab is not None:
        st.session_state.grab.stop()
        st.session_state.grab = None
    st.session_state.running = False
    logger.info("[UI] Streamlit: pipeline stopped.")

# Define the sidebar (Adjust if necessary)
def _ui_sidebar(cfg: AppConf, state: Optional[AppState]):
    st.sidebar.header("⚙️ Settings")

    # Device read-only display (Adjust via YAML/CLI if necessary)
    st.sidebar.text(f"Device: {get_device(cfg.device)}")
    st.sidebar.text(f"Backbone: {cfg.backbone}")
    st.sidebar.text(f"Text enc: {cfg.text_encoder}")

    # Retrieval and Decision hyperparameters
    st.sidebar.markdown("### Retrieval / Decision")
    cfg.retrieval.k = st.sidebar.slider("Top-K", min_value=1, max_value=20, value=cfg.retrieval.k, step=1)
    cfg.retrieval.alpha_fusion = st.sidebar.slider("α fusion (img vs text)", 0.0, 1.0, cfg.retrieval.alpha_fusion, 0.05)
    cfg.retrieval.temporal_ema = st.sidebar.slider("Temporal EMA", 0.0, 0.9, cfg.retrieval.temporal_ema, 0.05)
    cfg.decision.temperature = st.sidebar.slider("Temperature", 0.1, 2.0, cfg.decision.temperature, 0.05)
    cfg.decision.open_set_threshold = st.sidebar.slider("Open-set τ", 0.0, 1.0, cfg.decision.open_set_threshold, 0.01)

    # Assign the labels with the preloaded and updated text prototypes
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Labels")
    init_labels = st.sidebar.text_input("Preload labels (comma-separated)", value="bottle,cup,phone")
    if st.sidebar.button("Preload / Update Prototypes"):
        labels = [s.strip() for s in init_labels.split(",") if s.strip()]
        if state is not None:
            state.update_text_prototypes(labels)
            state.ensure_labels(labels)
            st.sidebar.success(f"Updated {len(labels)} label prototypes.")

    # Assign the few-shot registration from the recent buffers
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Registration")
    st.session_state.reg_label = st.sidebar.text_input("New class label", value="new_object")
    st.session_state.reg_frames = st.sidebar.slider("Frames to capture", 1, 32, 6, 1)
    st.session_state.reg_stride = st.sidebar.slider("Stride in buffer", 1, 8, 2, 1)
    if st.sidebar.button("Register from buffer"):
        if state is not None:
            n = state.register_from_buffer(st.session_state.reg_label,
                                           num_frames=st.session_state.reg_frames,
                                           stride=st.session_state.reg_stride)
            st.sidebar.success(f"Registered {n} vectors under '{st.session_state.reg_label}'")

    # Open-set toggle mirrors AppState flag
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Open-set")
    if state is not None:
        toggled = st.sidebar.checkbox("Enable open-set rejection (unknown)", value=state.open_set_enabled)
        state.open_set_enabled = toggled

    #  One-click snapshot of last frame
    st.sidebar.markdown("---")
    if st.sidebar.button("Save Snapshot"):
        if state is not None and "last_frame" in st.session_state and st.session_state.last_frame is not None:
            fn = os.path.join(state.snap_dir, f"st_{int(time.time())}.png")
            imwrite(fn, st.session_state.last_frame)
            st.sidebar.success(f"Saved {fn}")

# Construct the Streamlit layout and run the live render loop while the pipeline is active
def main():
    st.set_page_config(page_title="Vision-Fusion-RT", layout="wide")
    st.title("Vision-Fusion-RT — Streamlit")

    # Initialized the session objects
    _ensure_session()
    cfg: AppConf = st.session_state.cfg
    state: Optional[AppState] = st.session_state.state

    # Top controls
    start_col, stop_col, spacer = st.columns([1,1,6])
    with start_col:
        if st.button("▶️ Start", use_container_width=True):
            _start_pipeline()
            state = st.session_state.state
    with stop_col:
        if st.button("⏹ Stop", use_container_width=True):
            _stop_pipeline()

    # Sidebar and real time panels
    _ui_sidebar(cfg, state)
    left, right = st.columns([3,2])
    video_area = left.empty()
    metrics_area = right.container()
    log_area = right.container()

    # Status badges
    st.markdown("---")
    st.caption("Hotkeys in OpenCV window (if shown): [q] quit, [r] register, [s] snapshot, [o] open-set toggle, [t] text proto")

    # Declare the real time session based loop
    while st.session_state.running and st.session_state.grab is not None and st.session_state.state is not None:
        fr = st.session_state.grab.read_latest()
        if fr is None:
            time.sleep(0.003)
            continue
        
        # Retain the last frame for snapshots
        st.session_state.last_frame = fr.data

        # Declare the label, score, inference and overlay
        label, score, dt = st.session_state.state.infer(fr.data)
        fps = st.session_state.fps.step()
        vis = draw_overlay(fr.data.copy(), label=label, score=score, fps=fps)

        # Display the BGR into the RGB for the Streamlit UI
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        video_area.image(vis_rgb, caption=f"{label} ({score:.2f}) • {fps:.1f} FPS", use_column_width=True)

        # Declare the metrics and its region
        with metrics_area:
            col1, col2, col3 = st.columns(3)
            col1.metric("Label", label)
            col2.metric("Score", f"{score:.2f}")
            col3.metric("FPS", f"{fps:.1f}")
            st.progress(min(1.0, max(0.0, score)))

        # Set a time interval
        time.sleep(0.001)

    # Declare the idle hint
    if not st.session_state.running:
        st.info("Pipeline stopped. Click ▶️ Start to run.")

## ======================================================================================================
## DEMO AND TESTING
## ======================================================================================================
if __name__ == "__main__":
    main()

### ========================================================================================================================================
## END (ADD IMPLEMENTATIONS IF NECESSARY)
### ========================================================================================================================================