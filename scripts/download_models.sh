#!/usr/bin/env bash
#
# Vision-Fusion-RT — Offline Model Downloader
# -------------------------------------------
# Pre-fetches Hugging Face models/tokenizers/processors so first run is snappy.
#
# Usage:
#   scripts/download_models.sh \
#     --cache ./hf_cache \
#     --clip "openai/clip-vit-base-patch32" \
#     --timm "vit_base_patch16_224" \
#     --st "sentence-transformers/all-MiniLM-L6-v2" \
#     --extra "intfloat/e5-small-v2"
#
# Notes:
# - Prefers huggingface-cli when available; otherwise uses small Python snippets.
# - TIMM models are fetched by triggering a one-shot load in Python (cached in ~/.cache/torch/hub or HF cache).
# - You can run this multiple times; downloads are cached.

set -euo pipefail

# ---------- defaults ----------
CACHE_DIR="${HOME}/.cache/huggingface"
CLIP_MODEL="openai/clip-vit-base-patch32"
ST_MODEL="sentence-transformers/all-MiniLM-L6-v2"
EXTRA_MODELS=""            # comma-separated HF model ids
TIMM_MODEL=""              # e.g., vit_base_patch16_224

# ---------- helpers ----------
usage() {
  grep -E '^#' "$0" | sed -E 's/^# ?//'
}

exists() { command -v "$1" >/dev/null 2>&1; }

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache) CACHE_DIR="${2:-$CACHE_DIR}"; shift 2 ;;
    --clip)  CLIP_MODEL="${2:-$CLIP_MODEL}"; shift 2 ;;
    --st)    ST_MODEL="${2:-$ST_MODEL}"; shift 2 ;;
    --extra) EXTRA_MODELS="${2:-}"; shift 2 ;;
    --timm)  TIMM_MODEL="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[WARN] Unknown arg: $1"; shift ;;
  esac
done

mkdir -p "$CACHE_DIR"

# ---------- huggingface-cli path (optional) ----------
if exists huggingface-cli; then
  echo "[INFO] Using huggingface-cli (cache: $CACHE_DIR)"
  HF="huggingface-cli download --local-dir-use-symlinks False --local-dir"
else
  HF=""
  echo "[INFO] huggingface-cli not found; will use Python fallback."
fi

# ---------- download via huggingface-cli ----------
if [[ -n "$HF" ]]; then
  echo "[INFO] Downloading CLIP model: $CLIP_MODEL"
  $HF "$CACHE_DIR/$CLIP_MODEL" "$CLIP_MODEL" || true

  echo "[INFO] Downloading Sentence-Transformers model: $ST_MODEL"
  $HF "$CACHE_DIR/$ST_MODEL" "$ST_MODEL" || true

  if [[ -n "$EXTRA_MODELS" ]]; then
    IFS=',' read -ra EXS <<< "$EXTRA_MODELS"
    for M in "${EXS[@]}"; do
      M_TRIM="$(echo "$M" | xargs)"
      [[ -z "$M_TRIM" ]] && continue
      echo "[INFO] Downloading extra model: $M_TRIM"
      $HF "$CACHE_DIR/$M_TRIM" "$M_TRIM" || true
    done
  fi
fi

# ---------- Python fallback (and tokenizer/processor pulls) ----------
PY_FALLBACK=$(cat <<'PY'
import os, sys, json
cache_dir = os.environ.get("HF_CACHE_DIR", os.path.expanduser("~/.cache/huggingface"))
clip_model = os.environ.get("CLIP_MODEL")
st_model = os.environ.get("ST_MODEL")
extra = os.environ.get("EXTRA_MODELS", "")
from pathlib import Path

def fetch_clip(model_id: str, cache_dir: str):
    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast
    print(f"[PY] Fetching CLIP: {model_id}")
    CLIPModel.from_pretrained(model_id, cache_dir=cache_dir)
    CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    CLIPTokenizerFast.from_pretrained(model_id, cache_dir=cache_dir)

def fetch_st(model_id: str, cache_dir: str):
    try:
        from sentence_transformers import SentenceTransformer
        print(f"[PY] Fetching Sentence-Transformers: {model_id}")
        SentenceTransformer(model_id, cache_folder=cache_dir)
        return
    except Exception:
        pass
    from transformers import AutoModel, AutoTokenizer
    print(f"[PY] Fallback: transformers AutoModel/Tokenizer for {model_id}")
    AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
    AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

if clip_model:
    fetch_clip(clip_model, cache_dir)

if st_model:
    fetch_st(st_model, cache_dir)

if extra:
    for m in [s.strip() for s in extra.split(",") if s.strip()]:
        try:
            fetch_st(m, cache_dir)
        except Exception:
            try:
                fetch_clip(m, cache_dir)
            except Exception as e:
                print(f"[PY][WARN] Could not fetch {m}: {e}")

print("[PY] Done.")
PY
)

export HF_CACHE_DIR="$CACHE_DIR"
export CLIP_MODEL="$CLIP_MODEL"
export ST_MODEL="$ST_MODEL"
export EXTRA_MODELS="$EXTRA_MODELS"

python3 - <<"$PY_FALLBACK" || true
$PY_FALLBACK
PY_FALLBACK

# ---------- TIMM weights (optional) ----------
if [[ -n "$TIMM_MODEL" ]]; then
  echo "[INFO] Triggering TIMM weight download: $TIMM_MODEL"
  python3 - <<PY || true
import timm, torch
m = timm.create_model("${TIMM_MODEL}", pretrained=True)
m.eval()
x = torch.zeros(1,3,224,224)
with torch.no_grad():
    _ = m(x)
print("[PY] TIMM model loaded and ran once.")
PY
fi

echo "[OK] All requested models processed. Cache: $CACHE_DIR"
