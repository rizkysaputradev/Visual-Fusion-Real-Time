
# Vision‑Fusion‑RT — API Guide

This guide is **hybrid**: it begins with user‑facing tasks, then summarizes the main internal classes so contributors can extend the system confidently.

## Part I — User‑Facing API

### 1) Run the UI

```bash
scripts/run_streamlit.sh --config experiments/configs/default.yaml --device auto
```

Inside the UI you can:
- Watch live predictions.
- Register a new class (select frames or a folder).
- Calibrate open‑set threshold τ.
- Save / load the memory registry.

### 2) Programmatic Inference (Minimal)

```python
from src.core.config import load_config
from src.models.backbones.registry import build_backbone
from src.models.text.registry import build_text_encoder
from src.memory.incremental import IncrementalMemory
from src.retrieval.retriever import Retriever, RetrieverConfig
from src.models.heads.decision import DecisionHead
from src.pipeline.inference import InferenceEngine
from src.io.image_io import imread

cfg = load_config("experiments/configs/default.yaml")

img_enc = build_backbone(cfg.backbone.name, device=cfg.run.device, fp16=cfg.run.fp16)
txt_enc = build_text_encoder(cfg.text_encoder.name, device=cfg.run.device) if cfg.text_encoder.enabled else None

mem = IncrementalMemory(dim=img_enc.dim,
                        metric=cfg.memory.metric,
                        backend=cfg.memory.backend,
                        index_spec=cfg.memory.index_spec,
                        persist_dir=cfg.paths.persist_dir)

retr = Retriever(memory=mem,
                 image_encoder=img_enc,
                 text_encoder=txt_enc,
                 text_aligned=cfg.text_encoder.aligned,
                 cfg=RetrieverConfig(k=cfg.retrieval.k))

dec = DecisionHead(temperature=cfg.decision.temperature,
                   tau_open=cfg.decision.tau_open,
                   use_margin=cfg.decision.use_margin,
                   margin_delta=cfg.decision.margin_delta)

engine = InferenceEngine(retriever=retr, decision=dec)

frame = imread("data/samples/mug.jpg")
out = engine.infer(frame)
print(out.label, out.score, out.latency_ms)
```

### 3) Register a Class From a Folder
```python
from src.pipeline.register import register_from_folder
from src.models.preproc.augment import Augmenter

augmenter = Augmenter("mid")  # or "none"
added = register_from_folder(mem, img_enc, "data/samples/mug", label="mug", augmenter=augmenter)
print("vectors added:", added)
retr.update_text_prototypes(mem.registry.labels())
```

### 4) Save / Load Memory
```python
paths = mem.save_all(prefix="mem")
# Later
# mem = IncrementalMemory(...); mem.store = FaissStore.load(paths["index"]); mem.registry = LabelRegistry.load(paths["registry"])
```

---

## Part II — Internal Modules (Contributor‑level)

Below is a concise map of key classes with their contracts.

### Core

- **`core/config.py`**
  - `load_config(path) -> AppConf`: typed dataclass loader for YAML files.
- **`core/logging_utils.py`**
  - Structured logger + `PerfAccumulator`, `FPSMeter`.
- **`core/types.py`**
  - Small typed containers: `Frame`, `RetrievalResult`, etc.
- **`core/utils.py`**
  - Helpers: `seed_all`, `get_device`, `l2_normalize`, timers.

### IO

- **`io/webcam.py`**
  - `FrameGrabber(device_id, size, fps).start().read_latest().stop()`: threaded capture.
- **`io/image_io.py`**
  - `imread`, `imwrite`, `list_images`, `to_pil`, `to_bgr`.
- **`io/overlay.py`**
  - `draw_overlay(image, label, score, fps, show_threshold=None)`.

### Models — Vision Backbones

- **`models/backbones/clip_vision.py`**
  - Wrapper for CLIP image tower: `.encode_images(list, batch_size, tqdm=False) -> np.ndarray [N,d]` and `.dim`.
- **`models/backbones/vit_vision.py`**
  - Wrapper for ViT via `timm`/HF. Same interface as above.
- **`models/backbones/registry.py`**
  - `build_backbone(key, device="cpu", fp16=False, **kwargs)`; `list_backbones()`.

### Models — Text Encoders

- **`models/text/clip_text.py`**
  - CLIP text tower wrapper: `.encode_text(list, batch_size) -> [M,d]`.
- **`models/text/hf_text.py`**
  - Sentence‑Transformers / HF fallback encoders.
- **`models/text/registry.py`**
  - `build_text_encoder(key, device="cpu", **kwargs)`.

### Models — Preprocessing & Head

- **`models/preproc/transforms.py`**
  - Resize/normalize/center‑crop ops bound to encoders.
- **`models/preproc/augment.py`**
  - `Augmenter(strength).apply(pil_list, n_per_image=1) -> List[PIL]`.
- **`models/heads/decision.py`**
  - `DecisionHead(temperature, tau_open, use_margin, margin_delta)`:
    - `decide(labels, scores) -> (label, score)`
    - `fit_temperature(logits_list, targets)`
    - `auto_threshold(pos_scores, neg_scores, target_fpr)`

### Memory

- **`memory/schema.py`**
  - `LabelRegistry`: maps ids↔labels, stores `VectorMeta`.
- **`memory/faiss_store.py` / `memory/milvus_store.py`**
  - Backends exposing a unified API: `.add`, `.remove_ids`, `.search`, `.save`, `.load`.
- **`memory/store_router.py`**
  - `build_store(dim, backend="faiss", metric="ip", index_spec="Flat", **kwargs)`.
- **`memory/incremental.py`**
  - `IncrementalMemory`: orchestrates store + registry, supports `register_class`, `undo_last_for`, centroids, persistence.

### Retrieval

- **`retrieval/fusion.py`**
  - Math utilities: distance→similarity, neighbor aggregation, text fusion, top‑k selection.
- **`retrieval/retriever.py`**
  - High‑level: encode → search → aggregate → (optional) text fusion → EMA smoothing.
  - `RetrieverConfig(k, alpha_fusion, temporal_ema, neighbor_agg, text_prompt_template)`.

### Pipeline

- **`pipeline/encode.py`**
  - `encode_images_bgr`, `encode_texts`, `ensure_l2`.
- **`pipeline/register.py`**
  - Register new classes from frames/folders; set centroids; sync text prototypes.
- **`pipeline/inference.py`**
  - `InferenceEngine(retriever, decision).infer(bgr) -> InferenceOutput`.
- **`pipeline/calibrate.py`**
  - Temperature fitting + open‑set τ selection utilities.

### Eval

- **`eval/metrics.py`**
  - Top‑k accuracy, CMC, MRR, AP/mAP, ROC/EER, Brier, latency summaries.
- **`eval/bench_rt.py`**
  - Real‑time benchmark harness for webcam/video streams.

### Apps

- **`app/main_rt.py`**
  - Threaded loop (state, hotkeys, swap backends).
- **`app/ui_streamlit.py` / `app/ui_gradio.py`**
  - Minimal operator UI for demos and live registration.

---

## Extension Patterns

- **New backbone**: implement `encode_images` + `.dim`, add to backbones registry.
- **New text model**: implement `encode_text`, add to text registry; set `aligned=False` unless it’s CLIP‑space aligned.
- **Custom fusion**: add a function to `retrieval/fusion.py` and call it from `Retriever` via a config switch.
- **Scale up**: switch `memory.backend` to `milvus` in YAML and set connection options.

## Versioning & Repro

- Configs record: backbone name + model id, text encoder, memory spec, decision head params.
- Persisted artifacts (index/labels) live in `data/registries/` by default.