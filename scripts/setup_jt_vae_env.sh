#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${1:-$ROOT_DIR/.venv-jtvae}"
PYTHON_SPEC="${JT_VAE_PYTHON_SPEC:-3.10}"
BACKEND_DIR="$ROOT_DIR/src/modifications/ml_based/jt_vae/vendor/JTNN-VAE"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required for the preferred JT-VAE setup path." >&2
    echo "Install uv or create a dedicated conda environment and set JT_VAE_PYTHON manually." >&2
    exit 1
fi

if [ ! -d "$BACKEND_DIR" ]; then
    echo "JT-VAE backend checkout not found: $BACKEND_DIR" >&2
    echo "Run: git submodule update --init --recursive src/modifications/ml_based/jt_vae/vendor/JTNN-VAE" >&2
    exit 1
fi

uv venv --python "$PYTHON_SPEC" "$ENV_DIR"
# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"

uv pip install -r "$BACKEND_DIR/requirements.txt" tqdm
uv pip install -e "$BACKEND_DIR"

cat <<EOF
JT-VAE environment created at: $ENV_DIR

Export these variables before using JTVAEModifier:
  export JT_VAE_PYTHON=$ENV_DIR/bin/python
  export JT_VAE_HOME=$BACKEND_DIR
  export JT_VAE_VOCAB_PATH=$BACKEND_DIR/data/moses/vocab.txt
  export JT_VAE_MODEL_PATH=/absolute/path/to/pretrained/model.iter-XXXX
EOF
