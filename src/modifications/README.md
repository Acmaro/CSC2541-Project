## 4. JT-VAE (Junction Tree Variational Autoencoder)

**Location:** `ml_based/jt_vae/`
**Environment:** dedicated isolated environment (preferred setup: `bash scripts/setup_jt_vae_env.sh`)

JT-VAE is integrated through an isolated subprocess wrapper. The project-facing class lives in `ml_based/jt_vae/jtvae_modifier.py`, while the actual model execution happens in `ml_based/jt_vae/backend_infer.py` using a separate interpreter configured by `JT_VAE_PYTHON`.

This keeps the main project environment free from JT-VAE's dependency stack while preserving a simple `.modify()` interface for callers.

### Backend Layout

* **Vendored backend:** `ml_based/jt_vae/vendor/mol_opt/main/jt_vae/`
* **Wrapper:** `ml_based/jt_vae/jtvae_modifier.py`
* **Backend shim:** `ml_based/jt_vae/backend_infer.py`
* **Setup script:** `scripts/setup_jt_vae_env.sh`

### Required Runtime Assets

The JT-VAE integration expects the following runtime inputs:

```bash
export JT_VAE_PYTHON=/path/to/.venv-jtvae/bin/python
export JT_VAE_HOME=/path/to/src/modifications/ml_based/jt_vae/vendor/mol_opt/main/jt_vae
export JT_VAE_VOCAB_PATH=/path/to/src/modifications/ml_based/jt_vae/vendor/mol_opt/main/jt_vae/data/zinc/vocab.txt
export JT_VAE_MODEL_PATH=/path/to/src/modifications/ml_based/jt_vae/vendor/mol_opt/main/jt_vae/fast_molvae/vae_model/model.iter-25000
export JT_VAE_DEVICE=auto
```

Teammates must fetch the JT-VAE backend through the git submodule:

```bash
git submodule update --init --recursive
```

The checkpoint comes from the `mol_opt` submodule, not from Git LFS. The
vendored backend also includes compatible `moses` and `zinc` vocabulary
files.

### Usage

```python
from src.modifications.ml_based.jt_vae import JTVAEModifier

M = JTVAEModifier()
variants = M.modify("CCO", n=5)
```

### Implementation Notes

* The current integration uses seed-conditioned latent perturbation: it encodes the input molecule, perturbs the latent vectors with Gaussian noise, and decodes unique candidates.
* The backend shim prefers GPU automatically when CUDA is available and falls back to a CPU-safe execution path otherwise.
* The verified public checkpoint layout expects the `zinc` vocabulary in the vendored backend by default.
* If `uv` cannot resolve a working backend environment on a given machine, a dedicated conda environment can still be used as long as `JT_VAE_PYTHON` points to its interpreter.
