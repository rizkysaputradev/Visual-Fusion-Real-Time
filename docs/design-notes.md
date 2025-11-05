
# Vision‑Fusion‑RT — Design Notes
_A minimal, real‑time, few‑shot multimodal recognition system._

> Tone: clean, research‑grade, pragmatic. This document captures **why** each architectural choice was made, **how** components interact under real‑time constraints, and **what** trade‑offs we accept for latency, stability, and extensibility.

## 1. Objectives & Constraints

**Primary objective**: online recognition with **live memory extensibility** — new classes can be registered in seconds from a handful of frames, without end‑to‑end retraining.

**Key constraints**
- **Latency budget**: sub‑50 ms/frame on consumer GPUs for 640×480 (encode → retrieve → decide).
- **Few‑shot robustness**: tolerate 3–10 samples per class via augmentation + neighbor aggregation.
- **Open‑set safety**: abstain when uncertain; threshold learned from small dev sets.
- **Portability**: CPU‑capable (reduced FPS), GPU‑accelerated when available; minimal ops‑cost.
- **Determinism knobs**: per‑run seeds; consistent preprocessing; controlled augmenters.

## 2. System Overview

```
camera/video → preproc → vision encoder → embedding(q) → ANN store → neighbors
             ↘ (optional) text prompts → text encoder → prototypes
neighbors → per‑label aggregation → late fusion (image/text) → temporal EMA → decision head
                               ↑                         ↘ registration flow writes into store & registry
                           memory store (FAISS/Milvus)  + LabelRegistry (metadata, ids)
```

### Design intents
- **Separation of concerns**: encoders (image/text), retrieval, decision head, memory store, UIs are modular.
- **Late fusion**: image scores are dominant; text priors guide tie‑breaks and low‑shot regimes.
- **Vector DB pluggability**: FAISS (local, easy), Milvus (remote, scalable). Uniform API via `store_router`.
- **Idempotent registration**: `IncrementalMemory` assigns ids, writes to store, updates label metadata, and optionally centroids.

## 3. Encoders & Embedding Space

- **Vision**: CLIP ViT‑B/32 or ViT (timm/HF). Encoders emit **L2‑normalized** float32 embeddings.
- **Text**: CLIP text (aligned to image space) or semantic HF models (MiniLM/E5). If **unaligned**, we disable late fusion to avoid misleading scores.
- **Preprocessing**: CLIP uses its Processor; ViT uses standard resize‑center‑crop‑normalize. All preprocessing is isolated in `models/preproc`.

**Why CLIP‑first?** Mature, robust cross‑modal alignment; performs well in zero/few‑shot settings; ubiquitous weights.

## 4. Memory & Retrieval

- **Backends**: FAISS (Flat / IVF / IVF+PQ) for local prototyping; Milvus for at‑scale persistence and remote search.
- **IDs & metadata**: `LabelRegistry` keeps a bijection between vec‑ids and labels, plus per‑sample metadata (timestamp, source, path).
- **Retrieval path**: k‑NN/ANN → neighbor labels → **label‑level aggregation** (`max`/`mean`/`sum`) → optional **late fusion** with text prototypes.
- **Temporal smoothing**: EMA on top‑1 stabilizes flicker from noisy frames without hiding shifts.

**Trade‑offs**
- **Flat vs IVF/PQ**: Flat is exact and simple; IVF/PQ greatly reduces memory/search time for N≫100k at the cost of tiny accuracy loss. Configurable via YAML without touching code.
- **Centroids**: kept per‑label at registration time (mean of new batch). Exact recomputation on deletions is optional (add FAISS `.reconstruct(ids)` if needed).

## 5. Registration (Few‑Shot)

- **Batch or live**: register from folder (`scripts/import_folder_as_class.py`) or on‑the‑fly from frames.
- **Augmentations**: light photometric/geometry jitter improves robustness with tiny shot counts.
- **Id allocation**: monotonic ids; undo stack per label supports last‑N reverts.

**Why not incremental fine‑tuning?** Costly to do safely on‑device in real‑time; vector‑DB approach yields immediate class availability and keeps the model frozen — perfect for portfolio demos and pragmatic systems.

## 6. Decision Head & Open‑Set Handling

- **Temperature scaling**: converts fused scores into calibrated probabilities.
- **Open‑set threshold (τ)**: top‑1 probability must exceed τ; optional margin requires a gap over top‑2.
- **Calibration**: `pipeline/calibrate.py` estimates temperature on a small known dev set and τ via target FPR on unknowns.

**Philosophy**: Favor **precision** over recall in open‑world streams. It’s better to say “unknown” than to assert a wrong label repeatedly.

## 7. Real‑Time Loop & UI

- **Threaded capture** (`io/webcam.py`) decouples frame ingress from inference.
- **Streamlit/Gradio** UIs: register classes, watch predictions, set thresholds, export registries — minimal academic styling; dark theme; wide layout.
- **Overlay**: FPS, label, score, τ line.

## 8. Performance Notes

- **GPU**: CLIP‑B/32 encodes ~2–4 ms/frame on RTX‑class GPUs; FAISS Flat search O(100 µs – 1 ms) for memories ≤200k.
- **CPU**: expect single‑digit FPS; prefer smaller image sizes and Flat index.
- **Bottlenecks**: PIL↔ndarray conversions, non‑pinned host memory, excessive Streamlit callbacks. Use batch encode for offline ops.

## 9. Failure Modes & Mitigations

- **Label drift**: imprecise labels hurt neighbor aggregation → curate registration shots; use augmenters conservatively.
- **Domain shift**: poor lighting or motion blur → lower capture resolution, enable EMA, or register additional views.
- **Over‑confident openset**: calibrate τ on a small negative set and prefer `neighbor_agg=mean` for stability.

## 10. Extensibility Hooks

- Swap encoders in `models/backbones` and `models/text` registries.
- Add `FaissStore.reconstruct(ids)` if exact centroids are required post‑deletion.
- Implement active learning: log high‑entropy frames and prompt the operator to register them.
- Remote persistence: drop‑in Milvus config for cloud‑scale memories.

## 11. Security & Privacy

- No frames are persisted by default. Registries store **embeddings + metadata only**.
- If needed, enable on‑disk frame snapshots behind a clear opt‑in flag.

## 12. Reproducibility

- Seeds set at app start; deterministic transforms; fixed model versions recorded in logs.
- `experiments/configs/*.yaml` fully reproduces an app run when paired with a registry snapshot.

---

_This document is intentionally concise and balanced: strong enough for a grad‑level review, readable for engineers integrating the system live._
