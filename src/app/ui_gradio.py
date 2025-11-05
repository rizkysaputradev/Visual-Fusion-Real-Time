### ========================================================================================================================================
## Module       : src/app/ui_gradio.py
## Author       : Rizky Johan Saputra (Independent Project)
## Date         : 4th November 2025 (Seoul, South Korea)
## Project      : Vision Fusion Real Time System (Copyright 2025)
## Topics       : Computer Vision, Real-Time Systems, Interactive AI System, NLP, Machine Learning and Memory Augmentation
## Purpose      : Provide a browser-based Gradio interface to run and control Vision-Fusion-RT in real time:
##                start/stop streaming, visualize overlays (label/score/FPS), tune retrieval/decision
##                parameters, manage labels and few-shot registration from the buffer, and save snapshots.
## Role         : Gradio UI
### ========================================================================================================================================

## ======================================================================================================
## SPECIFICATIONS
## ======================================================================================================
"""
Vision-Fusion-RT — Gradio UI
----------------------------
Streaming webcam inference with controls for labels, registration, and thresholds.

Features
- Start/Stop live stream (generator yields frames)
- Display overlay image, current label/score/FPS
- Manage labels (preload/encode), registration from buffer
- Tune k, α fusion, temperature, open-set threshold, temporal EMA
- Save snapshots
"""

## ======================================================================================================
## SETUP (ADJUSTABLE) (ADJUST IF NECESSARY)
## ======================================================================================================
from __future__ import annotations
import os, time, threading
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import cv2

from src.core.config import load_config, AppConf
from src.core.logging_utils import logger, FPSMeter
from src.io.webcam import FrameGrabber
from src.io.overlay import draw_overlay
from src.io.image_io import imwrite
from src.app.main_rt import AppState

## ======================================================================================================
## IMPLEMENTATIONS
## ======================================================================================================
# Declare a global runtime and wrapped inside the launch to avoid side-effects during importing
class Runtime:
    def __init__(self):
        self.cfg: AppConf = load_config("experiments/configs/default.yaml")
        self.state: Optional[AppState] = None
        self.grab: Optional[FrameGrabber] = None
        self.fps = FPSMeter(beta=0.9)
        self.running: bool = False
        self.lock = threading.Lock()
        self.last_frame: Optional[np.ndarray] = None

    # Start the RT pipeline and camera safely (Idempotent under lock)
    def start(self):
        with self.lock:
            if self.state is None:
                self.state = AppState(self.cfg)
            if self.grab is None:
                self.grab = FrameGrabber(
                    device_id=self.cfg.rt.cam_device,
                    size=self.cfg.rt.cam_size,
                    fps=self.cfg.rt.cam_fps,
                    max_queue=self.cfg.rt.max_queue,
                    drop_on_backlog=self.cfg.rt.drop_frame_on_backlog
                ).start()
            self.running = True
            logger.info("[UI] Gradio: started")

    # Stop streaming and release camera resources (Safe to call in multiple times)
    def stop(self):
        with self.lock:
            self.running = False
            if self.grab is not None:
                self.grab.stop()
                self.grab = None
            logger.info("[UI] Gradio: stopped")

# Build and launch the Gradio UI
def launch():
    # Declare the runtime instance (Shared across the UI callbacks)
    R = Runtime()

    # Top-level Blocks layout
    with gr.Blocks(title="Vision-Fusion-RT") as demo:
        gr.Markdown("# Vision-Fusion-RT — Gradio")

        # Transport controls
        with gr.Row():
            start = gr.Button("▶️ Start", variant="primary")
            stop = gr.Button("⏹ Stop")
            snapshot = gr.Button("📸 Snapshot")

        # Real time image and fast diagnostics
        with gr.Row():
            image = gr.Image(label="Live", type="numpy", streaming=True)
            with gr.Column():
                label_box = gr.Textbox(label="Predicted Label", interactive=False)
                score_box = gr.Textbox(label="Score", interactive=False)
                fps_box = gr.Textbox(label="FPS", interactive=False)
                progress = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Confidence", interactive=False)

        # Retrieval and decision hyperparameters (Live-updated)
        gr.Markdown("### Retrieval / Decision")
        with gr.Row():
            k = gr.Slider(1, 20, value=R.cfg.retrieval.k, step=1, label="Top-K")
            alpha = gr.Slider(0.0, 1.0, value=R.cfg.retrieval.alpha_fusion, step=0.05, label="α fusion (img vs text)")
            ema = gr.Slider(0.0, 0.9, value=R.cfg.retrieval.temporal_ema, step=0.05, label="Temporal EMA")
        with gr.Row():
            temp = gr.Slider(0.1, 2.0, value=R.cfg.decision.temperature, step=0.05, label="Temperature")
            tau = gr.Slider(0.0, 1.0, value=R.cfg.decision.open_set_threshold, step=0.01, label="Open-set τ")
            open_set = gr.Checkbox(value=True, label="Enable open-set rejection")

        # Label management and buffer-based registration
        gr.Markdown("### Labels & Registration")
        with gr.Row():
            init_labels = gr.Textbox(value="bottle,cup,phone", label="Preload labels (comma-separated)")
            update_protos = gr.Button("Preload / Update Prototypes")
        with gr.Row():
            reg_label = gr.Textbox(value="new_object", label="New class label")
            reg_frames = gr.Slider(1, 32, value=6, step=1, label="Frames to capture")
            reg_stride = gr.Slider(1, 8, value=2, step=1, label="Stride in buffer")
            register_btn = gr.Button("Register from buffer", variant="secondary")

        # Start streaming
        def on_start():
            R.start()
            return gr.update(streaming=True)
        
        # Stop streaming
        def on_stop():
            R.stop()
            return gr.update(streaming=False)
        
        # Save the last displayed frame to disk
        def on_snapshot():
            if R.state and R.last_frame is not None:
                fn = os.path.join(R.state.snap_dir, f"gr_{int(time.time())}.png")
                imwrite(fn, R.last_frame)
                return f"Saved {fn}"
            return "No frame available."
        
        # Apply the slider and checkbox changes to the live configs and toggles
        def update_settings(kv, av, ema_v, temp_v, tau_v, open_v):
            R.cfg.retrieval.k = int(kv)
            R.cfg.retrieval.alpha_fusion = float(av)
            R.cfg.retrieval.temporal_ema = float(ema_v)
            R.cfg.decision.temperature = float(temp_v)
            R.cfg.decision.open_set_threshold = float(tau_v)
            if R.state is not None:
                R.state.open_set_enabled = bool(open_v)
            return gr.update()
        
        # Pre-encode text prototypes for comma-separated labels
        def on_update_protos(txt):
            labels = [s.strip() for s in (txt or "").split(",") if s.strip()]
            if R.state and labels:
                R.state.update_text_prototypes(labels)
                R.state.ensure_labels(labels)
                return f"Updated {len(labels)} labels."
            return "No labels provided."

        # Register few-shot samples from the frame buffer into memory
        def on_register(lbl, nf, stp):
            if R.state is None:
                return "State not ready."
            n = R.state.register_from_buffer(lbl or "new_object", int(nf), int(stp))
            return f"Registered {n} vectors to '{lbl}'."

        # Define a generator that yields frames while the system is running
        def streamer():
            while R.running and R.grab is not None and R.state is not None:
                fr = R.grab.read_latest()
                if fr is None:
                    time.sleep(0.003)
                    continue
                R.last_frame = fr.data
                label, score, dt = R.state.infer(fr.data)
                fps = R.fps.step()
                vis = draw_overlay(fr.data.copy(), label=label, score=score, fps=fps)

                # Convert the BGR into RGB for the Gradio UI
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                yield vis_rgb, label, f"{score:.2f}", f"{fps:.1f}", min(1.0, max(0.0, float(score)))

        # Wire UI events
        start.click(on_start, outputs=image)
        stop.click(on_stop, outputs=image)
        snapshot.click(on_snapshot, outputs=label_box)

        # Settings updates
        for ctrl in (k, alpha, ema, temp, tau, open_set):
            ctrl.change(update_settings, inputs=[k, alpha, ema, temp, tau, open_set], outputs=[])

        # Label ops
        update_protos.click(on_update_protos, inputs=init_labels, outputs=label_box)
        register_btn.click(on_register, inputs=[reg_label, reg_frames, reg_stride], outputs=label_box)

        # Streaming binding
        image.stream(streamer, outputs=[image, label_box, score_box, fps_box, progress])

    # Launch the demo
    demo.launch()

## ======================================================================================================
## DEMO AND TESTING
## ======================================================================================================
if __name__ == "__main__":
    launch()

### ========================================================================================================================================
## END (ADD IMPLEMENTATIONS IF NECESSARY)
### ========================================================================================================================================