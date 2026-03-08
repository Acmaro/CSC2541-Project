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
    ml_based/          # Lib-INVENT (encoder-decoder RNN scaffold decoration)
    rule_based/        # CReM (context-controlled fragment substitution)
  evaluation/          # AiZynthFinder retrosynthesis, QED/Tanimoto scoring
  utils/               # Shared preprocessing and featurization utilities
envs/                  # Conda environment specs (env_prexsyn.yml, env_libinvent.yml)
scripts/               # Setup and utility scripts
data/                  # Downloaded databases (e.g., CReM fragment DB)
notebooks/             # Exploratory notebooks
docs/                  # Project documentation
```

### Environments

| Tool | Environment | Python |
|------|-------------|--------|
| PrexSyn, AiZynthFinder, CReM | `prexsyn_env` (`envs/env_prexsyn.yml`) | 3.11 |
| Lib-INVENT | `libinvent_env` (`envs/env_libinvent.yml`) | 3.7 |

See `src/mutations/README.md` for full mutation pipeline usage.
