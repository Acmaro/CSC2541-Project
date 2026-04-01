"""
Validate PrexSyn synthesizability bias on a fresh 500-molecule ChEMBL sample.

Pipeline:
    ChEMBL (random 500) → PrexSyn → seed_smiles → AiZynthFinder → compare

Confirms the finding from seed_synthesizability.ipynb:
    PrexSyn output is more synthesizable than its ChEMBL input,
    especially for low-quality (large-modification) cases.

Outputs (all under OUT_DIR):
    prexsyn_500.json          — PrexSyn results (checkpoint, resumable)
    synth_500.json            — AiZynthFinder results (checkpoint, resumable)
    synth_bias_500_results.csv — per-molecule table
    synth_bias_500_summary.csv — summary by quality bin

Usage:
    conda run -n prexsyn_env python scripts/validate_prexsyn_synth_bias.py
    python scripts/validate_prexsyn_synth_bias.py --depth 3 --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, rdFingerprintGenerator, rdMolDescriptors
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.scoring_v2 import make_spec, tanimoto_to_spec
from src.evaluation.synth_parallel import worker_init, score_one

# ── Config ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',         type=int, default=500,   help='ChEMBL molecules to sample')
    p.add_argument('--samples',   type=int, default=256,   help='PrexSyn samples per molecule')
    p.add_argument('--depth',     type=int, default=6,     help='AiZynthFinder max depth')
    p.add_argument('--workers',   type=int, default=10,    help='AiZynthFinder parallel workers')
    p.add_argument('--time-limit',type=int, default=120,   help='AiZynthFinder timeout per molecule (s)')
    p.add_argument('--seed',      type=int, default=1234,  help='Random seed for ChEMBL sampling')
    p.add_argument('--url',       type=str, default='http://localhost:8011/sample',
                   help='PrexSyn API endpoint')
    return p.parse_args()


# ── Featurization (inline, matches featurize_chembl.py) ───────────────────────

_ECFP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_FCFP_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
)
_DESC_NAMES = list(rdMolDescriptors.Properties().GetPropertyNames())
MAX_BRICS   = 8


def _featurize_one(smi: str) -> dict | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    ecfp4 = _ECFP_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)
    fcfp4 = _FCFP_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)
    rdkit = np.nan_to_num(
        np.array(rdMolDescriptors.Properties(_DESC_NAMES).ComputeProperties(mol), dtype=np.float32),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    brics_fps    = np.zeros((MAX_BRICS, 2048), dtype=np.float32)
    brics_exists = np.zeros(MAX_BRICS, dtype=bool)
    try:
        frags = sorted(BRICS.BRICSDecompose(mol))[:MAX_BRICS]
    except Exception:
        frags = []
    for i, fsmi in enumerate(frags):
        fmol = Chem.MolFromSmiles(fsmi.replace('*', 'C'))
        if fmol:
            brics_fps[i]    = _ECFP_GEN.GetFingerprintAsNumPy(fmol).astype(np.float32)
            brics_exists[i] = True
    return {
        'ecfp4':  ecfp4, 'fcfp4': fcfp4, 'rdkit': rdkit,
        'brics':  brics_fps, 'bric_e': brics_exists,
    }


# ── Quality bin helper ─────────────────────────────────────────────────────────

def _quality_bin(t: float) -> str:
    if   t < 0.50: return '<0.5'
    elif t < 0.70: return '0.5-0.7'
    elif t < 0.85: return '0.7-0.85'
    else:          return '0.85-1.0'


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    GEN_DIR    = ROOT / 'data' / 'generation_stratified'
    OUT_DIR    = GEN_DIR / 'synth_bias_validation'
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    AIZSYNTH_CFG = ROOT / 'data' / 'aizynthfinder' / 'config.yml'
    assert AIZSYNTH_CFG.exists(), f'AiZynthFinder config missing: {AIZSYNTH_CFG}'

    PREXSYN_CKPT = OUT_DIR / f'prexsyn_{args.n}.json'
    SYNTH_CKPT   = OUT_DIR / f'synth_{args.n}_depth{args.depth}.json'
    RESULTS_CSV  = OUT_DIR / f'synth_bias_{args.n}_results.csv'
    SUMMARY_CSV  = OUT_DIR / f'synth_bias_{args.n}_summary.csv'

    BIN_ORDER = ['<0.5', '0.5-0.7', '0.7-0.85', '0.85-1.0']

    # ── Stage 1: sample ChEMBL molecules ──────────────────────────────────────
    print('=' * 60)
    print('Stage 1 — Sample ChEMBL molecules')

    existing_seeds = json.load(open(GEN_DIR / 'seeds_for_methods_stratified.json'))
    existing_specs = {s['spec_smiles'] for s in existing_seeds}

    chembl = pd.read_csv(ROOT / 'data' / 'chembl_full.csv')
    all_smiles = chembl['SMILES'].dropna().tolist()

    rng      = np.random.default_rng(args.seed)
    pool     = [s for s in all_smiles if s not in existing_specs]
    sampled  = [pool[i] for i in rng.choice(len(pool), size=min(args.n * 3, len(pool)), replace=False)]

    # Validate SMILES, keep first args.n valid
    valid: list[str] = []
    for smi in sampled:
        if len(valid) >= args.n:
            break
        if Chem.MolFromSmiles(smi) is not None:
            valid.append(smi)

    print(f'Sampled {len(valid)} valid ChEMBL molecules (excluded {len(existing_specs)} existing seeds)')

    # ── Stage 2: run PrexSyn ───────────────────────────────────────────────────
    print()
    print('=' * 60)
    print('Stage 2 — Run PrexSyn')

    prexsyn_results: list[dict] = []
    if PREXSYN_CKPT.exists():
        prexsyn_results = json.load(open(PREXSYN_CKPT))
        done_specs = {r['spec_smiles'] for r in prexsyn_results}
        print(f'Resumed: {len(prexsyn_results)} already done')
    else:
        done_specs = set()

    pending = [s for s in valid if s not in done_specs]
    print(f'Remaining: {len(pending)} molecules')

    api_errors = 0
    for smi in tqdm(pending, desc='PrexSyn', unit='mol'):
        feat = _featurize_one(smi)
        if feat is None:
            continue
        payload = {
            'ecfp4':             feat['ecfp4'].tolist(),
            'fcfp4':             feat['fcfp4'].tolist(),
            'rdkit_desc_values': feat['rdkit'].tolist(),
            'rdkit_desc_names':  _DESC_NAMES,
            'brics_fps':         feat['brics'].tolist(),
            'brics_exists':      feat['bric_e'].tolist(),
            'source_smiles':     smi,
            'num_samples':       args.samples,
        }
        try:
            resp       = requests.post(args.url, json=payload, timeout=300)
            resp.raise_for_status()
            candidates = resp.json().get('generated_smiles', [])
        except Exception as e:
            tqdm.write(f'  API error for {smi[:40]}: {e}')
            candidates = []
            api_errors += 1

        spec = make_spec(smi)
        best_smi, best_t = smi, 0.0
        if spec and candidates:
            for c in candidates:
                if '.' in c:
                    continue
                mol = Chem.MolFromSmiles(c)
                if mol:
                    t = tanimoto_to_spec(mol, spec)
                    if t > best_t:
                        best_t, best_smi = t, c

        prexsyn_results.append({
            'spec_smiles':      smi,
            'seed_smiles':      best_smi,
            'baseline_quality': round(best_t, 4),
            'quality_bin':      _quality_bin(best_t),
        })

        # Checkpoint every 50 molecules
        if len(prexsyn_results) % 50 == 0:
            with open(PREXSYN_CKPT, 'w') as f:
                json.dump(prexsyn_results, f)

    with open(PREXSYN_CKPT, 'w') as f:
        json.dump(prexsyn_results, f)
    print(f'PrexSyn done. API errors: {api_errors}. Saved -> {PREXSYN_CKPT.name}')

    from collections import Counter
    bin_counts = Counter(r['quality_bin'] for r in prexsyn_results)
    print('Quality bin distribution:')
    for b in BIN_ORDER:
        print(f'  {b:<10}: {bin_counts[b]:>4}')

    # ── Stage 3: run AiZynthFinder ─────────────────────────────────────────────
    print()
    print('=' * 60)
    print(f'Stage 3 — AiZynthFinder (depth {args.depth})')

    # Collect all unique SMILES: spec + seed
    all_to_score: set[str] = set()
    for r in prexsyn_results:
        all_to_score.add(r['spec_smiles'])
        all_to_score.add(r['seed_smiles'])
    all_to_score_list = list(all_to_score)
    print(f'Unique molecules to score: {len(all_to_score_list)} '
          f'(= {len(prexsyn_results)} specs + {len(prexsyn_results)} seeds - '
          f'{2*len(prexsyn_results) - len(all_to_score_list)} duplicates)')

    synth: dict[str, bool] = {}
    if SYNTH_CKPT.exists():
        synth = json.load(open(SYNTH_CKPT))
        print(f'Resumed: {len(synth)} already scored')

    pending_synth = [s for s in all_to_score_list if s not in synth]
    print(f'Remaining: {len(pending_synth)} molecules')

    timeout = args.time_limit + 30
    if pending_synth:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=worker_init,
            initargs=(str(AIZSYNTH_CFG), args.depth, args.time_limit),
            max_tasks_per_child=None,
        ) as pool_ex:
            futures = {pool_ex.submit(score_one, s): s for s in pending_synth}
            with tqdm(total=len(pending_synth), desc=f'AiZynth depth={args.depth}', unit='mol') as pbar:
                for fut in as_completed(futures):
                    smi = futures[fut]
                    try:
                        _, solved = fut.result(timeout=timeout)
                    except FutureTimeoutError:
                        solved = False
                    except BrokenProcessPool:
                        with open(SYNTH_CKPT, 'w') as f:
                            json.dump(synth, f)
                        raise
                    except Exception:
                        solved = False
                    synth[smi] = solved
                    pbar.update(1)

        with open(SYNTH_CKPT, 'w') as f:
            json.dump(synth, f)
        print(f'AiZynthFinder done. Saved -> {SYNTH_CKPT.name}')

    # ── Stage 4: results ───────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print('Stage 4 — Results')

    col = f'is_synth_depth{args.depth}'
    rows = []
    for r in prexsyn_results:
        rows.append({
            'spec_smiles':      r['spec_smiles'],
            'seed_smiles':      r['seed_smiles'],
            'baseline_quality': r['baseline_quality'],
            'quality_bin':      r['quality_bin'],
            'identity':         r['spec_smiles'] == r['seed_smiles'],
            f'spec_{col}':      synth.get(r['spec_smiles'], False),
            f'seed_{col}':      synth.get(r['seed_smiles'], False),
        })

    results_df = pd.DataFrame(rows)
    results_df['quality_bin'] = pd.Categorical(results_df['quality_bin'], categories=BIN_ORDER, ordered=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f'Per-molecule results saved -> {RESULTS_CSV.name}')

    # Summary table
    summary_rows = []
    for b in BIN_ORDER + ['Overall']:
        sub = results_df if b == 'Overall' else results_df[results_df['quality_bin'] == b]
        n   = len(sub)
        if n == 0:
            continue
        spec_synth = sub[f'spec_{col}'].sum()
        seed_synth = sub[f'seed_{col}'].sum()
        summary_rows.append({
            'Quality Bin':           b,
            'N':                     n,
            'ChEMBL synth':          int(spec_synth),
            'ChEMBL synth (%)':      round(spec_synth / n * 100, 1),
            'PrexSyn synth':         int(seed_synth),
            'PrexSyn synth (%)':     round(seed_synth / n * 100, 1),
            'Δ (PrexSyn - ChEMBL)':  round((seed_synth - spec_synth) / n * 100, 1),
            'Identity (%)':          round(sub['identity'].mean() * 100, 1),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print()
    print(f'Synthesizability across pipeline layers  (AiZynthFinder depth {args.depth}, n={len(results_df)})\n')
    print(summary_df.to_string(index=False))
    print()

    # Compare with seed_synthesizability.ipynb finding (n=100)
    overall = summary_df[summary_df['Quality Bin'] == 'Overall'].iloc[0]
    ref_chembl = 60.0
    ref_prexsyn = 70.0
    print(f'Reference (n=100 stratified): ChEMBL={ref_chembl}%  PrexSyn={ref_prexsyn}%  Δ=+10.0pp')
    print(f'This run  (n={len(results_df):>3} random):    '
          f'ChEMBL={overall["ChEMBL synth (%)"]:.1f}%  '
          f'PrexSyn={overall["PrexSyn synth (%)"]:.1f}%  '
          f'Δ={overall["Δ (PrexSyn - ChEMBL)"]:+.1f}pp')


if __name__ == '__main__':
    main()
