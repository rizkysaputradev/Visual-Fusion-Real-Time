### ========================================================================================================================================
## Module       : src/core/config.py
## Author       : Rizky Johan Saputra (Independent Project)
## Date         : 4th November 2025 (Seoul, South Korea)
## Project      : Vision Fusion Real Time System (Copyright 2025)
## Topics       : Computer Vision, Real-Time Systems, Interactive AI System, NLP, Machine Learning and Memory Augmentation
## Purpose      : Provide a typed, YAML-driven configuration system for Vision-Fusion-RT: define nested
##                dataclasses (app/memory/retrieval/decision/rt), load & validate configs, support layered
##                overrides (YAML → Python dict → environment), and offer save/serialize helpers.
## Role         : Configuration System
### ========================================================================================================================================

## ======================================================================================================
## SPECIFICATIONS
## ======================================================================================================
"""
Vision-Fusion-RT — Configuration System
---------------------------------------

Goals
- Strongly-typed configuration via dataclasses (no runtime magic).
- YAML loader with layered overrides:
    1) YAML file (base)
    2) Python dict overrides (e.g., from CLI/UI)
    3) Environment variables (optional; prefixed 'VFRT_' per field)
- Validation and normalization (e.g., ensuring tuples, clamping ranges).
- Save/Load helpers for reproducibility.
- Small utilities for convenient in-code overrides (apply_overrides).

Design
- The configuration is broken into nested dataclasses:
  AppConf -> (MemoryConf, RetrievalConf, DecisionConf, RTConf)
- Each conf object can be serialized (to_dict) and reloaded.

Notes
- Keep this module torch-agnostic. Device strings are handled here as text; actual device
  objects are created in `core.utils.get_device`.
- This module should not import heavy libraries.
"""

## ======================================================================================================
## SETUP (ADJUSTABLE) (ADJUST IF NECESSARY)
## ======================================================================================================
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Any, Dict, Tuple
import os
import yaml

## ======================================================================================================
## IMPLEMENTATIONS
## ======================================================================================================
# Defining the dataclass for each config domain, loaders, validators, and override utilities
@dataclass
class MemoryConf:
    """Memory and cache for the running program"""
    backend: Literal["faiss", "milvus"] = "faiss"
    metric: Literal["ip", "l2"] = "ip"

    # Index/ANN specifics
    index_spec: str = "Flat"
    nprobe: int = 16
    enforce_normalize: bool = False
    train_threshold: int = 200

    # Persistence options (Revert to YAML)
    persist: bool = True                  
    persist_dir: str = "data/registries"  

    # Declare the back-compat and sanity normalization
    def __post_init__(self):
        # Set the normalization and limits
        self.nprobe = int(max(1, int(self.nprobe)))
        self.train_threshold = int(max(0, int(self.train_threshold)))
        self.enforce_normalize = bool(self.enforce_normalize)
        if self.metric not in ("ip", "l2"):
            self.metric = "ip"
        if not isinstance(self.index_spec, str) or not self.index_spec.strip():
            self.index_spec = "Flat"

        # Ensure to assign the right directory if issue remains persisting
        if self.persist and self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)

# Declare the dataclass for the retrieval configuration
@dataclass
class RetrievalConf:
    """Retrieval of data and metrics"""
    # Define the number of neighbors
    k: int = 5

    # Assign the neighbor aggregation over K
    neighbor_agg: Literal["mean", "max", "sum", "median", "softmax"] = "mean"
    neighbor_temp: float = 1.0  # used when neighbor_agg == "softmax"

    # Assign the normalization of candidate scores
    score_norm: Literal["none", "minmax", "zscore", "softmax"] = "none"

    # Allocate the fusion strategy between the image and text branches
    fuse: Literal["none", "late", "early"] = "late"

    # Assign the fusion weights (Alpha or explicit, adjust if necessary)
    alpha_fusion: float = 0.5
    weight_image: float = 0.5
    weight_text: float = 0.5

    # Temporal smoothing of live scores (EMA over frames)
    temporal_ema: float = 0.0  # 0 = off, else e.g. 0.2

    # Declare its housekeeping
    dedup: bool = True
    filter_self: bool = True

    # Define the initialization
    def __post_init__(self):
        # Assing the basic clamps
        self.k = max(1, int(self.k))
        if self.neighbor_agg not in ("mean", "max", "sum", "median", "softmax"):
            self.neighbor_agg = "mean"
        try:
            self.neighbor_temp = float(self.neighbor_temp)
        except Exception:
            self.neighbor_temp = 1.0

        # Validate the normalized scores and fusion
        if self.score_norm not in ("none", "minmax", "zscore", "softmax"):
            self.score_norm = "none"
        if self.fuse not in ("none", "late", "early"):
            self.fuse = "late"

        # Assing the clamp for alpha and EMA
        try:
            self.alpha_fusion = float(self.alpha_fusion)
        except Exception:
            self.alpha_fusion = 0.5
        self.alpha_fusion = 0.0 if self.alpha_fusion < 0 else (1.0 if self.alpha_fusion > 1 else self.alpha_fusion)
        try:
            self.temporal_ema = float(self.temporal_ema)
        except Exception:
            self.temporal_ema = 0.0
        self.temporal_ema = max(0.0, min(0.99, self.temporal_ema))

        # Reconcile the explicit weights with alpha
        wi = float(self.weight_image)
        wt = float(self.weight_text)
        if wi <= 0 and wt <= 0:
            wi = self.alpha_fusion
            wt = 1.0 - self.alpha_fusion
        s = wi + wt
        if self.fuse != "none":
            if s == 0:
                wi, wt = 0.5, 0.5
            else:
                wi, wt = wi / s, wt / s
        self.weight_image, self.weight_text = wi, wt

# Declare the dataclass for the decision configuration
@dataclass
class DecisionConf:
    """Final decision head, calibration, and open-set behavior."""
    open_set_threshold: float = 0.28
    temperature: float = 0.9
    enable_platt: bool = False
    enable_temper: bool = True

# Declare the dataclass for the real time basis configuration
@dataclass
class RTConf:
    """Real-time camera and UI behavior."""
    max_queue: int = 4
    drop_frame_on_backlog: bool = True
    overlay: bool = True
    cam_device: int = 0
    cam_size: Tuple[int, int] = (640, 480)  # (W, H)
    cam_fps: int = 30

# Declare the dataclass for the app configuration
@dataclass
class AppConf:
    """Top-level configuration."""
    # Set the device source ("mps" and "cpu" is another option, adjust if necessary)
    device: str = "cuda"                 
    seed: int = 42
    backbone: str = "clip_vit_b32"            
    text_encoder: str = "clip_text"          
    img_size: Tuple[int, int] = (224, 224)
    normalize: str = "clip"                  
    memory: MemoryConf = field(default_factory=MemoryConf)
    retrieval: RetrievalConf = field(default_factory=RetrievalConf)
    decision: DecisionConf = field(default_factory=DecisionConf)
    rt: RTConf = field(default_factory=RTConf)

    # Serialize to a plain dictionary (YAML/JSON-friendly)
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # Declare the basic validation and normalization pass
    def validate(self) -> "AppConf":
        self.retrieval.k = int(max(1, self.retrieval.k))
        self.retrieval.alpha_fusion = float(min(1.0, max(0.0, self.retrieval.alpha_fusion)))
        self.retrieval.temporal_ema = float(min(0.99, max(0.0, self.retrieval.temporal_ema)))
        self.decision.temperature = float(max(0.01, self.decision.temperature))
        self.decision.open_set_threshold = float(min(1.0, max(0.0, self.decision.open_set_threshold)))

        # Check for sanity in the memory configuration
        self.memory.nprobe = int(max(1, self.memory.nprobe))
        self.memory.train_threshold = int(max(0, self.memory.train_threshold))
        self.memory.enforce_normalize = bool(self.memory.enforce_normalize)
        if self.memory.metric not in ("ip", "l2"):
            self.memory.metric = "ip"
        if not isinstance(self.memory.index_spec, str) or not self.memory.index_spec.strip():
            self.memory.index_spec = "Flat"

        # Ensure the persisting directory exists
        if self.memory.persist_dir:
            os.makedirs(self.memory.persist_dir, exist_ok=True)

        # Normalize the camera size into (W, H) size (Adjust if necessary)
        w, h = self.rt.cam_size
        self.rt.cam_size = (int(w), int(h))
        self.rt.cam_fps = int(max(1, self.rt.cam_fps))
        self.seed = int(self.seed)

        # Ensure the persisting directory exists
        if self.memory.persist_dir:
            os.makedirs(self.memory.persist_dir, exist_ok=True)
        return self

# Recursively merge the dictionary `sup` into dictionary `base` due to priority
def _merge_dict(base: Dict[str, Any], sup: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (sup or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out

# Construct a dataclass from a section of a dictionary
def _build_dc(dc, section: str, data: Dict[str, Any]):
    # Ensure the inclusion aliases and filtering on the unknown keys for richer YAML
    val = data.get(section, {})
    if not isinstance(val, dict):
        return dc()

    # Define shallow copy for edits
    val = {**val}

    # Set a condition in the memory section
    if section == "memory":
        # Set index for index_spec
        if "index_spec" not in val and "index" in val:
            val["index_spec"] = val.pop("index")

        # Assign the persisiting directory aliases
        for a in ("persist_path", "path", "dir"):
            if "persist_dir" not in val and a in val:
                val["persist_dir"] = val.pop(a)

        # Set the persisting string into enable and set directory
        if "persist" in val and isinstance(val["persist"], str):
            if "persist_dir" not in val:
                val["persist_dir"] = val["persist"]
            val["persist"] = True

    # Set a condition in the retrieval section
    elif section == "retrieval":
        # Set agg for neighbor agg
        if "neighbor_agg" not in val and "agg" in val:
            val["neighbor_agg"] = val.pop("agg")

        # Set fusion for fuse
        if "fuse" not in val and "fusion" in val:
            val["fuse"] = val.pop("fusion")

        # Set the dictionary weights for flat
        if "weights" in val and isinstance(val["weights"], dict):
            w = val.pop("weights")
            if "weight_image" not in val and "image" in w:
                val["weight_image"] = w["image"]
            if "weight_text" not in val and "text" in w:
                val["weight_text"] = w["text"]

    # Set a condition in the decision section
    elif section == "decision":
        # Set the open threshold for the open set threshold (Retain the YAML key)
        if "open_set_threshold" not in val and "tau_open" in val:
            val["open_set_threshold"] = val.pop("tau_open")

    # Filter to fields that the dataclass actually has (Prevents future crashes)
    allowed = set(getattr(dc, "__dataclass_fields__", {}).keys())
    val = {k: v for k, v in val.items() if k in allowed}
    return dc(**val)

# Declare the environment variable overrider into the nested dataclasses
def _apply_env_overrides(cfg: AppConf, prefix: str = "VFRT_") -> AppConf:
    # Initialize the flat dictionary
    flat: Dict[str, str] = {}
    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue

        # Strip the prefix
        key = k[len(prefix):] 
        flat[key] = v

    # Assign the navigation with the object and data path
    def assign(obj, path: str, raw: str):
        # Navigate the nested objects and cast based on existing field type
        parts = path.split("__")
        target = obj
        for p in parts[:-1]:
            target = getattr(target, p.lower())
        leaf = parts[-1].lower()
        old = getattr(target, leaf)

        # Declare the inference type from the existing field
        if isinstance(old, bool):
            val = str(raw).lower() in ("1", "true", "yes", "on")
        elif isinstance(old, int):
            val = int(raw)
        elif isinstance(old, float):
            val = float(raw)
        elif isinstance(old, tuple):

            # Instantiate the comma-separated values
            toks = [t.strip() for t in str(raw).split(",")]
            if all(tok.isdigit() for tok in toks):
                val = tuple(int(tok) for tok in toks)
            else:
                val = tuple(float(tok) for tok in toks)
        else:
            val = raw
        setattr(target, leaf, val)

    # Declare the presence of the key and raw data in the flat dictionary
    for key, raw in flat.items():
        try:
            assign(cfg, key, raw)
        except Exception:
            # Ignore the malformed env overrides (Logging is also approachable)
            pass
    return cfg

# Load the app configurations with YAML and apply a Python dictionary overrides
def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None,
                apply_env: bool = True) -> AppConf:
    data: Dict[str, Any] = {}
    if path:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

    if overrides:
        data = _merge_dict(data, overrides)

    # Normalize registry keys that may appear as dicts
    bval = data.get("backbone", "clip_vit_b32")
    if isinstance(bval, dict):
        backbone_name = bval.get("name", "clip_vit_b32")
    else:
        backbone_name = str(bval)

    tval = data.get("text_encoder", "clip_text")
    if isinstance(tval, dict):
        textenc_name = tval.get("name", "clip_text")
    else:
        textenc_name = str(tval)

    # Assign the parameterization in the app based configurations
    cfg = AppConf(
        device=data.get("device", "cuda"),
        seed=int(data.get("seed", 42)),
        backbone=backbone_name,      
        text_encoder=textenc_name,       
        img_size=tuple(data.get("img_size", [224, 224])),
        normalize=data.get("normalize", "clip"),
        memory=_build_dc(MemoryConf, "memory", data),
        retrieval=_build_dc(RetrievalConf, "retrieval", data),
        decision=_build_dc(DecisionConf, "decision", data),
        rt=_build_dc(RTConf, "rt", data),
    )
    if apply_env:
        cfg = _apply_env_overrides(cfg)

    return cfg.validate()

# Serialize the app configuration into the YAML file on disk
def save_config(cfg: AppConf, path: str) -> None:
    """Save the configuration to YAML."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)

# Declare a simple Python overrider (e.g., from argparse or UI)
def apply_overrides(cfg: AppConf, **kwargs) -> AppConf:
    """
    Apply simple Python overrides (e.g., from an argparse namespace).
    Example:
        cfg = apply_overrides(cfg, device="cpu", seed=123, rt={'cam_fps': 60})
    """
    for k, v in kwargs.items():
        if not hasattr(cfg, k):
            continue
        cur = getattr(cfg, k)
        if isinstance(cur, (MemoryConf, RetrievalConf, DecisionConf, RTConf)) and isinstance(v, dict):
            for kk, vv in v.items():
                if hasattr(cur, kk):
                    setattr(cur, kk, vv)
        else:
            setattr(cfg, k, v)
    return cfg.validate()

### ========================================================================================================================================
## END (ADD IMPLEMENTATIONS IF NECESSARY)
### ========================================================================================================================================