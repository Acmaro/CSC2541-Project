# Team Development & Architecture Guidelines (Project: Expanding Synthesizable Chemical Space)

This repository implements a multi-stage computational drug discovery pipeline: Generation (PrexSyn) -> Mutation (Lib-INVENT, CReM) -> Evaluation (AiZynthFinder, Scoring).

## 1. Core Principles & Workflow (Anthropic Standard)
The AI assistant must act as a senior staff engineer and adhere to the following workflows:
* **Plan First:** For any non-trivial task (3+ steps or architectural decisions), enter **Plan Mode** first to outline the approach. Verify the plan before implementation.
* **Verification Before Done:** Never mark a task complete without proving it works. Run tests, check logs, or dry-run scripts to demonstrate correctness.
* **Self-Improvement Loop:** If the user corrects a mistake, update internal lessons to prevent repeating it.
* **Simplicity First:** Make every change as simple and elegant as possible. Impact minimal code. No temporary or hacky fixes.

## 2. Strict Directory Structure
When creating or modifying files, strictly place them in the appropriate directory under `src/`. Do not clutter the root directory.
* `src/generation/`: Generative models (e.g., PrexSyn, Docker API).
* `src/mutations/`: Post-generation modification algorithms.
  * `ml_based/`: ML-based algorithms (e.g., Lib-INVENT, JT-VAE).
  * `rule_based/`: Rule-based algorithms (e.g., CReM, mmpdb).
* `src/evaluation/`: Evaluation modules (Tanimoto, QED scoring, AiZynthFinder retrosynthesis).
* `src/utils/`: Preprocessing, featurization, and common utilities.

## 3. Environment Awareness
This repository contains mixed execution environments. Always check `envs/` before running commands.
* **PrexSyn / AiZynthFinder:** Python 3.11 (`envs/env_prexsyn.yml`).
* **Lib-INVENT:** Python 3.7 (`envs/env_libinvent.yml`). Requires RDKit, PyTorch 1.7, openeye-toolkits.
* Do not introduce breaking changes across versions or forcefully integrate incompatible libraries. Use `subprocess` or API calls to bridge environments if necessary.

## 4. Sub-Module Architecture Context
* **Lib-INVENT (`src/mutations/ml_based/Lib-INVENT/`):**
  * Core model is an encoder-decoder RNN (`DecoratorModel`) that takes a scaffold SMILES with attachment points (e.g., `[*:0]`, `[*:1]`) and generates decorations.
  * Execution is driven by JSON configs passed to `input.py`, which dispatches to managers based on `"run_type"` (e.g., `scaffold_decorating`, `reinforcement_learning`).
* **PrexSyn API (`src/generation/docker/`):**
  * A FastAPI service wrapping PrexSyn. Expected to expose endpoints like `POST /predict` or `/sample` taking SMILES/features and returning analogs.
* **AiZynthFinder (`src/evaluation/retrosynthesis/`):**
  * Requires valid configuration and data files (e.g., `uspto_model.onnx`, valid HDF5 stock files) to perform Monte Carlo Tree Search for synthesis planning.

## 5. Communication & Language
* All documentation (README, code comments, and this CLAUDE.md) must be written in **English** so all team members can read them.
* Git commit messages must also be written in **English**, clearly explaining the intent behind the changes.
* Any drafts, notes, or messages intended for the team must default to **English**.