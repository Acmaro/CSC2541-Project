"""
prexsyn_reproducibility.py
──────────────────────────
Run PrexSyn twice on the same 100 seed molecules and compute the Jaccard
similarity between the two output sets.  This directly tests whether the
saturation claim is valid: if Jaccard ≈ 1, the output space is highly
reproducible (finite / small), supporting the saturation argument.
If Jaccard ≈ 0, the model is stochastic with a large sample space.

Metric definitions
──────────────────
Per-seed Jaccard:
    J_i = |run1_i ∩ run2_i| / |run1_i ∪ run2_i|
    where run1_i and run2_i are the sets of unique SMILES generated for
    seed molecule i in each run.

Global Jaccard:
    J_global = |∪_i run1_i  ∩  ∪_i run2_i| / |∪_i run1_i  ∪  ∪_i run2_i|

Usage
─────
    python scripts/prexsyn_reproducibility.py
    python scripts/prexsyn_reproducibility.py --num-samples 64 --url http://...
    python scripts/prexsyn_reproducibility.py --out data/generation_stratified/reproducibility.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_NPZ  = pathlib.Path("data/generation_stratified/chembl_features_stratified.npz")
DEFAULT_URL  = "http://100.65.172.100:8011/sample"
DEFAULT_N    = 64       # samples per seed per run
DEFAULT_OUT  = pathlib.Path("data/generation_stratified/reproducibility.json")
TIMEOUT      = 120      # seconds per request
MAX_RETRIES  = 3


# ── API call ──────────────────────────────────────────────────────────────────

def call_prexsyn(payload: dict[str, Any], url: str) -> list[str]:
    """
    POST payload to PrexSyn /sample and return list of generated SMILES.
    Returns empty list on failure.
    """
    for attempt in range(MAX_RETRIES):
        try:
            req  = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=TIMEOUT).read())
            return resp.get("generated_smiles", [])
        except urllib.error.HTTPError as e:
            print(f"    HTTP {e.code}: {e.read().decode()[:80]}", flush=True)
            break
        except Exception as e:
            wait = 2 ** attempt
            print(f"    attempt {attempt + 1}/{MAX_RETRIES} failed: {e}  (retry in {wait}s)",
                  flush=True)
            time.sleep(wait)
    return []


# ── Jaccard ───────────────────────────────────────────────────────────────────

def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0          # both empty — vacuously identical
    union = a | b
    return len(a & b) / len(union)


# ── main ──────────────────────────────────────────────────────────────────────

def run(npz: pathlib.Path, url: str, num_samples: int, out: pathlib.Path) -> None:
    # Load seed features
    data             = np.load(npz, allow_pickle=True)
    smiles_arr       = data["smiles"]
    ecfp4_arr        = data["ecfp4"]
    fcfp4_arr        = data["fcfp4"]
    rdkit_vals_arr   = data["rdkit_desc_values"]
    rdkit_names      = data["rdkit_desc_names"].tolist()
    brics_fps_arr    = data["brics_fps"]
    brics_exists_arr = data["brics_exists"]

    n_seeds = len(smiles_arr)
    print(f"Loaded {n_seeds} seed molecules from {npz}")
    print(f"Running two independent PrexSyn passes ({num_samples} samples/seed each)...")
    print(f"Endpoint: {url}\n")

    per_seed: list[dict] = []
    run1_global: set[str] = set()
    run2_global: set[str] = set()

    for i in range(n_seeds):
        seed_smi = str(smiles_arr[i])
        payload  = {
            "ecfp4":             ecfp4_arr[i].tolist(),
            "fcfp4":             fcfp4_arr[i].tolist(),
            "rdkit_desc_values": rdkit_vals_arr[i].tolist(),
            "rdkit_desc_names":  rdkit_names,
            "brics_fps":         brics_fps_arr[i].tolist(),
            "brics_exists":      brics_exists_arr[i].tolist(),
            "source_smiles":     seed_smi,
            "num_samples":       num_samples,
        }

        print(f"[{i+1:>3}/{n_seeds}] {seed_smi[:55]}", flush=True)

        run1 = set(call_prexsyn(payload, url))
        run2 = set(call_prexsyn(payload, url))

        j    = jaccard(run1, run2)
        only1 = len(run1 - run2)
        only2 = len(run2 - run1)
        both  = len(run1 & run2)

        print(f"         run1={len(run1):>3}  run2={len(run2):>3}  "
              f"shared={both:>3}  only-run1={only1:>3}  only-run2={only2:>3}  "
              f"Jaccard={j:.3f}", flush=True)

        per_seed.append({
            "seed_smiles": seed_smi,
            "run1_unique": len(run1),
            "run2_unique": len(run2),
            "shared":      both,
            "only_run1":   only1,
            "only_run2":   only2,
            "jaccard":     round(j, 4),
        })

        run1_global |= run1
        run2_global |= run2

    # Global Jaccard across all seeds pooled
    j_global    = jaccard(run1_global, run2_global)
    j_per_seed  = [r["jaccard"] for r in per_seed]

    summary = {
        "n_seeds":           n_seeds,
        "num_samples_per_run": num_samples,
        "url":               url,
        "jaccard_per_seed": {
            "mean":   round(float(np.mean(j_per_seed)), 4),
            "median": round(float(np.median(j_per_seed)), 4),
            "std":    round(float(np.std(j_per_seed)), 4),
            "min":    round(float(np.min(j_per_seed)), 4),
            "max":    round(float(np.max(j_per_seed)), 4),
        },
        "jaccard_global":    round(j_global, 4),
        "run1_total_unique": len(run1_global),
        "run2_total_unique": len(run2_global),
        "shared_global":     len(run1_global & run2_global),
        "per_seed":          per_seed,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved full results → {out}")

    # ── Print summary report ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("REPRODUCIBILITY REPORT")
    print("=" * 60)
    print(f"Seeds tested           : {n_seeds}")
    print(f"Samples per seed/run   : {num_samples}")
    print()
    print("Per-seed Jaccard similarity:")
    print(f"  Mean   : {summary['jaccard_per_seed']['mean']:.4f}")
    print(f"  Median : {summary['jaccard_per_seed']['median']:.4f}")
    print(f"  Std    : {summary['jaccard_per_seed']['std']:.4f}")
    print(f"  Min    : {summary['jaccard_per_seed']['min']:.4f}")
    print(f"  Max    : {summary['jaccard_per_seed']['max']:.4f}")
    print()
    print(f"Global Jaccard (pooled): {summary['jaccard_global']:.4f}")
    print(f"Run 1 unique SMILES    : {summary['run1_total_unique']:,}")
    print(f"Run 2 unique SMILES    : {summary['run2_total_unique']:,}")
    print(f"Shared (both runs)     : {summary['shared_global']:,}")
    print()

    # Interpretation
    j_med = summary["jaccard_per_seed"]["median"]
    print("INTERPRETATION:")
    if j_med >= 0.80:
        print(f"  Median Jaccard {j_med:.3f} ≥ 0.80 → HIGH reproducibility.")
        print("  PrexSyn's output space is small and largely deterministic per seed.")
        print("  The saturation claim is well-supported.")
    elif j_med >= 0.50:
        print(f"  Median Jaccard {j_med:.3f} in [0.50, 0.80] → MODERATE reproducibility.")
        print("  Some stochasticity exists, but the output space is still constrained.")
        print("  The saturation claim holds partially; soften the wording in the paper.")
    else:
        print(f"  Median Jaccard {j_med:.3f} < 0.50 → LOW reproducibility.")
        print("  PrexSyn's sampling is highly stochastic with a large output space.")
        print("  The saturation claim is NOT supported. Revise the paper argument.")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Test PrexSyn output reproducibility via two independent runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz",         type=pathlib.Path, default=DEFAULT_NPZ,
                   help="NPZ features file for the 100 seed molecules")
    p.add_argument("--url",         type=str,          default=DEFAULT_URL,
                   help="PrexSyn /sample endpoint")
    p.add_argument("--num-samples", type=int,          default=DEFAULT_N,
                   help="Samples to request per seed per run")
    p.add_argument("--out",         type=pathlib.Path, default=DEFAULT_OUT,
                   help="Output JSON path for full results")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if not args.npz.exists():
        print(f"ERROR: NPZ file not found: {args.npz}", file=sys.stderr)
        sys.exit(1)

    # Quick API reachability check before committing to 200 API calls
    print(f"Checking API reachability: {args.url} ...", flush=True)
    try:
        ping = urllib.request.urlopen(
            args.url.replace("/sample", "/health"), timeout=5
        )
        print(f"API is up: {ping.read().decode()[:60]}\n")
    except Exception as e:
        print(f"WARNING: API health check failed ({e})")
        print("Proceeding anyway — each call will retry up to 3 times.\n")

    run(npz=args.npz, url=args.url, num_samples=args.num_samples, out=args.out)
