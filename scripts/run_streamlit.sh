#!/usr/bin/env bash
#
# Vision-Fusion-RT — Streamlit launcher
# -------------------------------------
# Bootstraps a virtualenv (if missing), installs deps, and launches the Streamlit UI.
#
# Usage:
#   scripts/run_streamlit.sh \
#     --config experiments/configs/default.yaml \
#     --host 0.0.0.0 --port 8501 \
#     --device auto \
#     --venv .venv \
#     --ui wide
#
# Notes:
# - Extra args after `--` are forwarded to the Streamlit app (src/app/ui_streamlit.py).
# - Sets PYTHONPATH to repo root so you can run without installing as a package.

set -euo pipefail

# ---------- defaults ----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
APP_FILE="src/app/ui_streamlit.py"

PORT="8501"
HOST="localhost"
DEVICE="auto"           # auto|cuda|mps|cpu
CONFIG="experiments/configs/default.yaml"
VENV_DIR=".venv"
UI_LAYOUT="wide"        # wide|centered
EXTRA_APP_ARGS=()

# ---------- helpers ----------
usage() {
  grep -E '^#' "$0" | sed -E 's/^# ?//'
  echo
  echo "Extra app args can be passed after --, e.g.:"
  echo "  scripts/run_streamlit.sh -- --debug"
}

exists() { command -v "$1" >/dev/null 2>&1; }

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)   PORT="${2:-8501}"; shift 2 ;;
    --host)   HOST="${2:-localhost}"; shift 2 ;;
    --device) DEVICE="${2:-auto}"; shift 2 ;;
    --config) CONFIG="${2:-experiments/configs/default.yaml}"; shift 2 ;;
    --venv)   VENV_DIR="${2:-.venv}"; shift 2 ;;
    --ui)     UI_LAYOUT="${2:-wide}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_APP_ARGS=("$@"); break ;;
    *) echo "[WARN] Unknown arg: $1"; shift ;;
  esac
done

# ---------- sanity ----------
if [[ ! -f "$REPO_ROOT/$APP_FILE" ]]; then
  echo "[ERR] Streamlit app not found at: $REPO_ROOT/$APP_FILE" >&2
  exit 1
fi
mkdir -p "$REPO_ROOT"

# ---------- python & venv ----------
if ! exists python3; then
  echo "[ERR] python3 is required." >&2; exit 1
fi

PYTHON=python3
if [[ ! -d "$REPO_ROOT/$VENV_DIR" ]]; then
  echo "[INFO] Creating virtualenv at $VENV_DIR ..."
  "$PYTHON" -m venv "$REPO_ROOT/$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$REPO_ROOT/$VENV_DIR/bin/activate"

# ---------- install deps ----------
REQ_TXT="$REPO_ROOT/requirements.txt"
PYPROJECT="$REPO_ROOT/pyproject.toml"

if [[ -f "$REQ_TXT" ]]; then
  echo "[INFO] Installing requirements.txt ..."
  pip install --upgrade pip
  pip install -r "$REQ_TXT"
elif [[ -f "$PYPROJECT" ]]; then
  echo "[INFO] Installing from pyproject.toml ..."
  pip install --upgrade pip
  pip install -e "$REPO_ROOT"
else
  echo "[WARN] No requirements.txt or pyproject.toml found. Attempting bare run."
fi

# ---------- envs ----------
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"
export VFR_DEVICE="$DEVICE"                # consumed by ui_streamlit if desired
export VFR_UI_LAYOUT="$UI_LAYOUT"
export VFRT_CONFIG="$CONFIG"


# ---------- torch device hint (optional) ----------
if [[ "$DEVICE" == "auto" ]]; then
  # best-effort hints for CUDA/MPS
  if "$PYTHON" - <<'PY'
import torch, sys
dev = "cpu"
if torch.cuda.is_available():
    dev = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    dev = "mps"
print(dev)
PY
  then :; fi
fi

# ---------- launch ----------
cd "$REPO_ROOT"
echo "[INFO] Launching Streamlit on http://$HOST:$PORT with config=$CONFIG"
exec streamlit run "$APP_FILE" \
  --server.headless true \
  --server.address "$HOST" \
  --server.port "$PORT" \
  --client.showErrorDetails true \
  --browser.gatherUsageStats false \
  --theme.base "dark" \
  -- \
  --config "$CONFIG" \
  --device "$DEVICE" \
  --ui "$UI_LAYOUT" \
  "${EXTRA_APP_ARGS[@]}"
