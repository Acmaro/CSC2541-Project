# CSC2541 Project — Expanding Synthesizable Chemical Space

A multi-stage computational drug discovery pipeline:

**Generation (PrexSyn)** → **Mutation (Lib-INVENT, CReM)** → **Evaluation (AiZynthFinder, Scoring)**

---

### Git LFS Setup

This repository uses **Git Large File Storage (LFS)** to manage large binary model files (`*.model`, `*.empty`). Without LFS configured, cloning will give you small pointer files instead of the actual model weights.

**Why LFS?**
Lib-INVENT ships pre-trained model files (~90 MB each). Storing large binaries directly in git bloats history and slows clones. LFS stores the content on a separate server and keeps only lightweight pointers in the repository.

**Install Git LFS** (one-time, per machine):

```bash
# macOS
brew install git-lfs

# Ubuntu / Debian
sudo apt install git-lfs

# Conda
conda install -c conda-forge git-lfs
```

**After cloning**, run:

```bash
git lfs install   # register LFS hooks in this repo
git lfs pull      # download actual model files (replaces pointer files)
```

---

### Repository Layout

```
src/
  generation/          # PrexSyn generative model & Docker API
  mutations/
    ml_based/          # Lib-INVENT (implemented), JT-VAE (planned)
    rule_based/        # CReM (implemented), mmpdb (planned)
  evaluation/          # AiZynthFinder retrosynthesis, QED/Tanimoto scoring
  utils/               # Shared preprocessing and featurization utilities
envs/                  # Conda environment specs (env_prexsyn.yml, env_libinvent.yml)
scripts/               # Setup and utility scripts
data/                  # Downloaded databases (e.g., CReM fragment DB)
notebooks/             # Exploratory notebooks
docs/                  # Project documentation
```

### Mutation Modules

| Tool | Type | Status | Environment |
|------|------|--------|-------------|
| Lib-INVENT | ML-based (encoder-decoder RNN) | Implemented | `libinvent_env` (Python 3.7) |
| CReM | Rule-based (fragment substitution) | Implemented | `prexsyn_env` (Python 3.11) |
| mmpdb | Rule-based (matched molecular pairs) | Planned | `prexsyn_env` (Python 3.11) |
| JT-VAE | ML-based (junction tree VAE) | Planned | TBD |

**mmpdb** will complement CReM by providing property-directed transforms derived from matched molecular pair (MMP) analysis. Where CReM substitutes fragments based on structural context, mmpdb encodes activity cliffs and property changes observed across compound series — making it well-suited for guided optimization toward a target property profile.

**JT-VAE** (Junction Tree Variational Autoencoder) will provide latent-space perturbation as an alternative to discrete fragment operations. By encoding molecules into a continuous latent space structured around junction trees, it enables smooth interpolation between molecular scaffolds — complementary to the rule-based and RL-based approaches already in the pipeline.

### Environments

| Tool | Environment | Python |
|------|-------------|--------|
| PrexSyn, AiZynthFinder, CReM, mmpdb | `prexsyn_env` (`envs/env_prexsyn.yml`) | 3.11 |
| Lib-INVENT | `libinvent_env` (`envs/env_libinvent.yml`) | 3.7 |
| JT-VAE | TBD | TBD |

See `src/mutations/README.md` for full mutation pipeline usage and implementation details.
