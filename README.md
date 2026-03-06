# CSC2541 — PrexSyn Benchmarking Pipeline

End-to-end benchmarking of [PrexSyn](https://github.com/luost26/prexsyn), a transformer-based model for generating synthesizable drug-like molecules. Given a source molecule, the pipeline generates analogs conditioned on its molecular properties and scores them for structural similarity, drug-likeness, and novelty.

## Overview

```
ChEMBL download → filter → sample → featurize → PrexSyn API → score → results
```

| Stage | Script | Description |
|---|---|---|
| Download | `download_chembl.py` | Download full ChEMBL, filter valid organic molecules |
| Sample | `download_chembl.py` | Randomly sample N molecules from the full set |
| Featurize | `featurize_chembl.py` | Compute ECFP4, FCFP4, RDKit descriptors, BRICS fragments |
| Generate | `sampler.py` | Batch-call PrexSyn `/sample` API for all molecules |
| Score | `scoring.py` | Tanimoto similarity, desirability, hit classification |

## Setup

```bash
conda env create -f environment.yml
conda activate csc2541
```

The PrexSyn inference server must be running as a Docker container on port 8011:

```bash
docker start prexsyn
```

## Usage

### Jupyter Notebook (recommended)

Open `pipeline.ipynb` and run cells top to bottom. Edit the config cell at the top to adjust paths and parameters:

```python
NUM_MOLECULES = 1000   # molecules to benchmark
SEED          = 42     # sampling seed
NUM_SAMPLES   = 64     # analogs to generate per molecule
LIMIT         = None   # set to 10 for a quick test
```

The download step (`chembl_full.csv`) only needs to run once. To benchmark a different subset, change `SEED` and re-run the sample cell — no re-download required.

### Command Line

Run individual steps:

```bash
# 1. Download full ChEMBL and save all valid molecules (~1.8M)
python main.py download

# 2. Sample a subset for benchmarking
python main.py sample-chembl --num-molecules 1000 --seed 42

# 3. Compute molecular features
python main.py featurize

# 4. Generate analogs via PrexSyn API
python main.py sample --num-samples 64

# 5. Score results
python main.py score
```

Or run the full pipeline in one command:

```bash
python main.py all --num-molecules 1000 --num-samples 64
```

## Data Files

| File | Description |
|---|---|
| `data/chembl_full.csv` | All valid filtered ChEMBL molecules (SMILES column) |
| `data/chembl_sampled.csv` | Sampled subset used for this benchmark run |
| `data/chembl_1k_features.npz` | Pre-computed molecular feature arrays |
| `data/chembl_sampled.json` | Raw API responses with generated SMILES |
| `data/chembl_scores.csv` | Per-molecule scoring results |

### Feature Array Shapes (`chembl_1k_features.npz`)

| Key | Shape | Description |
|---|---|---|
| `smiles` | `(N,)` | Canonical SMILES strings |
| `ecfp4` | `(N, 2048)` | ECFP4 Morgan fingerprint |
| `fcfp4` | `(N, 2048)` | FCFP4 feature-based fingerprint |
| `rdkit_desc_values` | `(N, 43)` | RDKit scalar descriptors |
| `rdkit_desc_names` | `(43,)` | Descriptor names (index → name) |
| `brics_fps` | `(N, 8, 2048)` | ECFP4 of top-8 BRICS fragments |
| `brics_exists` | `(N, 8)` | Fragment slot mask |

## Scoring

Generated molecules are scored on three dimensions:

- **Tanimoto similarity** — ECFP4 similarity to the source molecule
- **Desirability** — composite score: `QED × Lipinski_penalty × MW_penalty × RotBond_penalty`
- **Hit classification** — a molecule is a *hit* if it is similar (Tanimoto ≥ 0.3), drug-like (desirability ≥ 0.2), and novel (Tanimoto < 1.0)

## Generation

The `/sample` endpoint uses `QuerySampler` with a product-of-experts AND query over all five property types (ECFP4, FCFP4, RDKit descriptors, RDKit upper bound, BRICS fragments). This is the principled inference method recommended by the PrexSyn authors for conditional generation.
