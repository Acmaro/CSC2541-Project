# JT-VAE Integration

This module integrates JT-VAE as an isolated post-generation molecular
modification backend.

## Architecture

JT-VAE is executed through a project-owned subprocess wrapper instead of
importing its backend directly into the main project environment.

This follows the same architectural principle already used in this
repository for dependency-sensitive tools:
- keep the project-facing API small and stable
- isolate legacy or conflicting dependencies in a separate environment
- exchange data through explicit file paths and subprocess calls

The wrapper class is `jtvae_modifier.py`, and the backend shim executed in
the isolated environment is `backend_infer.py`.

## Backend Choice

For feasibility, this integration vendors the Python 3 JTNN-VAE fork from
`croningp/JTNN-VAE` as a git submodule under `vendor/JTNN-VAE`.

The backend still requires a separate environment because it depends on a
different PyTorch/RDKit stack from the main project environment.

## Environment Setup

The preferred setup path uses `uv`:

```bash
bash scripts/setup_jt_vae_env.sh
```

This creates `.venv-jtvae/` and installs the vendored JTNN-VAE backend in
editable mode.

After setup, export the runtime variables:

```bash
export JT_VAE_PYTHON="$(pwd)/.venv-jtvae/bin/python"
export JT_VAE_HOME="$(pwd)/src/modifications/ml_based/jt_vae/vendor/JTNN-VAE"
export JT_VAE_VOCAB_PATH="$(pwd)/src/modifications/ml_based/jt_vae/vendor/JTNN-VAE/data/moses/vocab.txt"
export JT_VAE_MODEL_PATH=/absolute/path/to/pretrained/model.iter-XXXX
```

If `uv` cannot resolve a working torch/RDKit combination on your machine,
the fallback is to create a dedicated conda environment for the same
backend and point `JT_VAE_PYTHON` to that interpreter.

## Usage

```python
from src.modifications.ml_based.jt_vae import JTVAEModifier

modifier = JTVAEModifier()
variants = modifier.modify(
    "CCc1nn(C)c2ccc(cc12)C(=O)NCc3ccccc3",
    n=20,
)
```

## Runtime Notes

- The wrapper validates the backend checkout, interpreter, vocabulary, and
  checkpoint path before execution.
- The vendored backend contains hardcoded `.cuda()` calls, so the backend
  shim forces a CPU-safe path for reliability.
- The current implementation performs seed-conditioned latent perturbation:
  it encodes the input molecule, adds Gaussian noise in latent space, and
  decodes unique candidate molecules.
- Pretrained checkpoints are not shipped with this repository and must be
  provided separately.
