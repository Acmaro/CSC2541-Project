# Post-Generation Molecular Modification Module

This module implements the second stage of the drug discovery pipeline, which generates structural variants from initial synthesizable candidates. Modification algorithms are organized into two categories based on their computational approach and environment requirements.

## Overview

**ML-based approaches** (Python 3.7 environment) employ trained neural networks for scaffold decoration and latent-space exploration. **Rule-based approaches** (Python 3.11 environment) apply fragment substitution using transformation databases derived from chemical corpora.

Different tools have conflicting dependencies, so careful environment management is required before execution. Each tool is accessed through a simple `.modify(smiles, n)` interface that returns a list of generated variants.

---

## 1. Lib-INVENT (ML-based)

**Location:** `ml_based/Lib-INVENT/`
**Environment:** Python 3.7 (`envs/env_libinvent.yml`)
**Interface:** `LibINVENTModifier.modify(seed_smiles, n) -> list[str]`

Lib-INVENT performs reaction-based scaffold decoration by taking molecules with marked attachment points (e.g., `[*:0]`, `[*:1]`) and generating decorative extensions. The core model is an encoder-decoder RNN (`DecoratorModel`) that fine-tunes toward target fingerprints using reinforcement learning.

### Architecture

* **Core:** encoder-decoder RNN with attachment-point decoration
* **Pipeline:** fragment input molecules → fine-tune toward target fingerprints → rank outputs
* **Execution:** JSON configs passed to `input.py`, dispatches to managers based on `"run_type"` (e.g., `scaffold_decorating`, `reinforcement_learning`)

### The Fingerprint-Modification Pipeline

We have built an end-to-end pipeline (`ml_based/pipeline/`) that takes an input SMILES, fragments it, fine-tunes Lib-INVENT toward a target fingerprint via RL, and generates new molecules.

* **`fragment.py`**: Decomposes input via BRICS/RECAP, keeps fragments with 1–3 attachment points and ≥5 heavy atoms, and renumbers to `[*:0]`, `[*:1]`.
* **`configs.py`**: Builds JSON configs for Lib-INVENT. Uses `tanimoto_similarity` against the target SMILES (with optional QED weight). Applies DAP learning strategy + IdenticalMurckoScaffold diversity filter.
* **`run.py`**: CLI entry point that orchestrates Lib-INVENT subprocess calls and ranks output.

### Usage Examples

Run these commands from the project root with the appropriate environment active:

```bash
# Basic usage: modify input SMILES toward target fingerprint
python src/modifications/ml_based/pipeline/run.py \
    --input  "CCc1nn(C)c2ccc(cc12)C(=O)NCc3ccccc3" \
    --target "CCc1nn(C)c2ccc(cc12)C(=O)NCCN3CCOCC3" \
    --run-dir runs/test

# Skip RL (test fragmentation + decoration only from pre-trained model)
python src/modifications/ml_based/pipeline/run.py --input "..." --target "..." --skip-rl

# Adjust RL intensity
python src/modifications/ml_based/pipeline/run.py --input "..." --target "..." --n-steps 500 --batch 128

# Add QED as a secondary objective (weight 0.3 alongside Tanimoto weight 1.0)
python src/modifications/ml_based/pipeline/run.py --input "..." --target "..." --qed 0.3
```

Outputs go into `--run-dir` and include: `scaffolds.smi`, `rl_config.json`, `decorate_config.json`, `rl_model.pt`, and `decorated.csv`.

### Core Lib-INVENT Commands

If you need to interact directly with the Lib-INVENT core (run from inside `src/modifications/ml_based/Lib-INVENT/`):

```bash
# Run any mode via JSON config (scaffold_decorating, reinforcement_learning, etc.)
python input.py <path/to/config.json>

# Run unit tests
python main_test.py
```

---

## 2. CReM (Rule-based, ChEMBL/ZINC)

**Location:** `rule_based/crem_modifier.py`
**Environment:** Python 3.11 (main project environment)
**Interface:** `CRemModifier.modify(seed_smiles, n) -> list[str]`
**Database:** `data/crem_db/chembl22_sa2_hac12.db` (~1.1 GB)

CReM queries a pre-built ChEMBL/ZINC database containing matched molecular pair (MMP) transformations. It performs context-aware fragment swaps using SMARTS pattern matching, enabling rapid rule-based variant generation without neural network inference.

### Architecture

* **Transformation source:** ChEMBL/ZINC matched molecular pair database
* **Mechanism:** context-aware fragment swaps using SMARTS patterns
* **Advantage:** no model training required; fixed, interpretable transformation space

### Database Setup

Download the transformation database (~2 GB) from the CReM Zenodo release and place it at a path of your choice:

```
https://zenodo.org/record/4519690   ->  replacements02_sc2.db
```

Provide the path either directly to the constructor or via the `CREM_DB_PATH` environment variable:

```bash
export CREM_DB_PATH=/data/crem_db/replacements02_sc2.db
```

Test CReM installation:
```bash
python scripts/test_crem.py
```

### Usage

```python
from src.modifications.rule_based.crem_modifier import CRemModifier

# Construct once; reuse for many molecules
M = CRemModifier(db_path="data/crem_db/replacements02_sc2.db")

# Generate up to 50 canonical SMILES variants
variants = M.modify("CCc1nn(C)c2ccc(cc12)C(=O)NCc3ccccc3", n=50)
# variants: list[str] — up to 50 canonical SMILES strings
```

Or, relying on the environment variable:

```python
M = CRemModifier()   # reads CREM_DB_PATH automatically
variants = M.modify("c1ccccc1", n=20)
```

### Key Parameters

| Parameter  | Default | Description |
|------------|---------|-------------|
| `db_path`  | `None`  | Path to SQLite transformation DB (falls back to `CREM_DB_PATH`) |
| `radius`   | `3`     | Context SMARTS match radius. Higher = more conservative modifications |
| `min_size` | `0`     | Min heavy atoms in the replaced fragment |
| `max_size` | `10`    | Max heavy atoms in the replaced fragment |
| `min_inc`  | `-3`    | Min change in heavy atom count |
| `max_inc`  | `3`     | Max change in heavy atom count |

---

## 3. mmpdb (Rule-based, Custom Corpus)

**Location:** `rule_based/mmpdb_modifier.py` (planned implementation)
**Environment:** Python 3.11 (main project environment)
**Interface:** `MMPDBModifier.modify(seed_smiles, n) -> list[str]`

Unlike CReM's fixed ChEMBL database, mmpdb constructs custom transformation databases from reference corpora. This gives full control over the transformation space: you supply the corpus, mmpdb derives the rules. This is useful when you want transformations derived from your own proprietary chemical library or a specific domain.

### Architecture

* **Transformation source:** user-supplied molecular corpus (e.g., ChEMBL sample, in-house library)
* **Mechanism:** matched molecular pair indexing from your data
* **Advantage:** customizable transformation space; no external database dependency

### Setup

mmpdb requires 20–40 minutes to index a ChEMBL corpus sample. Setup uses environment variables for database path configuration, enabling reusable instances across multiple molecules.

### Usage

```python
from src.modifications.rule_based.mmpdb_modifier import MMPDBModifier

# Index your corpus once (one-time, takes 20-40 min for ChEMBL)
M = MMPDBModifier(db_path="path/to/custom.mmpdb", build_from_corpus="path/to/molecules.smi")

# Reuse the indexed database for many molecules
variants = M.modify("CCO", n=5)
```

---

## 4. JT-VAE (ML-based, Junction Tree VAE)

**Location:** `ml_based/jt_vae/`
**Environment:** Dedicated isolated environment (setup: `bash scripts/setup_jt_vae_env.sh`)

JT-VAE is integrated through an isolated subprocess wrapper. The project-facing class lives in `jtvae_modifier.py`, while the actual model execution happens in `backend_infer.py` using a separate interpreter configured by `JT_VAE_PYTHON`.

This design keeps the main project environment free from JT-VAE's dependency stack while maintaining a simple `.modify()` interface for callers.

### Architecture

* **Approach:** seed-conditioned latent perturbation
* **Process:** encode input molecule → perturb latent vectors with Gaussian noise → decode unique candidates
* **Acceleration:** GPU-enabled when CUDA is available; CPU fallback supported

### Backend Layout

* **Vendored backend:** `ml_based/jt_vae/vendor/mol_opt/main/jt_vae/`
* **Wrapper class:** `ml_based/jt_vae/jtvae_modifier.py`
* **Backend shim:** `ml_based/jt_vae/backend_infer.py`
* **Setup script:** `scripts/setup_jt_vae_env.sh`

### Required Runtime Configuration

```bash
export JT_VAE_PYTHON=/path/to/.venv-jtvae/bin/python
export JT_VAE_HOME=/path/to/src/modifications/ml_based/jt_vae/vendor/mol_opt/main/jt_vae
export JT_VAE_VOCAB_PATH=/path/to/src/modifications/ml_based/jt_vae/vendor/mol_opt/main/jt_vae/data/zinc/vocab.txt
export JT_VAE_MODEL_PATH=/path/to/src/modifications/ml_based/jt_vae/vendor/mol_opt/main/jt_vae/fast_molvae/vae_model/model.iter-25000
export JT_VAE_DEVICE=auto
```

Fetch the backend via git submodule:

```bash
git submodule update --init --recursive
```

The checkpoint and vocabulary files come from the `mol_opt` submodule, not Git LFS.

### Usage

```python
from src.modifications.ml_based.jt_vae import JTVAEModifier

M = JTVAEModifier()
variants = M.modify("CCO", n=5)
```

### Implementation Notes

* Encodes input, applies Gaussian perturbation to latent vectors, then decodes candidates
* Automatically prefers GPU when CUDA is available; falls back to CPU-safe execution
* Expected to use the `zinc` vocabulary in the vendored backend
* If `uv` cannot resolve a working environment, a dedicated conda environment works as long as `JT_VAE_PYTHON` points to its interpreter

---

## Common Interface

All modification tools follow the same pattern:

```python
modifier = SomeModifier(config_args)
variants = modifier.modify(input_smiles, n=num_variants)  # returns list[str]
```

**Choose tools based on your needs:**
- **Lib-INVENT:** Training-based decorator; good for scaffold variations with fingerprint targeting
- **CReM:** Fast rule-based lookup; fixed ChEMBL/ZINC transformation space
- **mmpdb:** Fast rule-based lookup; customizable transformation space from your own corpus
- **JT-VAE:** Learned latent-space exploration; diverse molecular variants without explicit rules
