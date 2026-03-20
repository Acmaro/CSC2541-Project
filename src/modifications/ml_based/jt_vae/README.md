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

For feasibility and checkpoint compatibility, this integration vendors the
`fast_jtnn` implementation from `wenhao-gao/mol_opt/main/jt_vae` under
`vendor/mol_opt/main/jt_vae/`.

This backend matches the public `model.iter-25000` checkpoint layout and
its default `zinc` vocabulary. The backend still runs in a separate
environment because it depends on a different PyTorch/RDKit stack from the
main project environment.

## Environment Setup

The preferred setup path uses `uv`:

```bash
bash scripts/setup_jt_vae_env.sh
```

This creates `.venv-jtvae/` and installs the JT-VAE runtime dependencies.
The backend source itself is vendored in this repository.

After setup, export the runtime variables:

```bash
export JT_VAE_PYTHON="$(pwd)/.venv-jtvae/bin/python"
export JT_VAE_HOME="$(pwd)/src/modifications/ml_based/jt_vae/vendor/mol_opt/main/jt_vae"
export JT_VAE_VOCAB_PATH="$(pwd)/src/modifications/ml_based/jt_vae/vendor/mol_opt/main/jt_vae/data/zinc/vocab.txt"
export JT_VAE_MODEL_PATH=/absolute/path/to/pretrained/model.iter-XXXX
export JT_VAE_DEVICE=auto
```

If `uv` cannot resolve a working torch/RDKit combination on your machine,
the fallback is to create a dedicated conda environment for the same
backend and point `JT_VAE_PYTHON` to that interpreter.

## Usage

```python
from src.modifications.ml_based.jt_vae import JTVAEModifier

modifier = JTVAEModifier()
variants = modifier.modify("CCO", n=5)
```

## Runtime Notes

- The wrapper validates the backend checkout, interpreter, vocabulary, and
  checkpoint path before execution.
- The backend shim prefers GPU automatically when CUDA is available and
  falls back to a CPU-safe path when it is not.
- The current implementation performs seed-conditioned latent perturbation:
  it encodes the input molecule, perturbs the latent vectors with Gaussian
  noise, and decodes unique candidate molecules.
- Legacy probabilistic decode paths are not the default because the old
  backend can fail inside `torch.multinomial` on some trees.
- Pretrained checkpoints are not shipped with this repository and must be
  provided separately.
