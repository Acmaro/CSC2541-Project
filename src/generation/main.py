"""
CSC2541 Project — PrexSyn Benchmarking Pipeline

Subcommands:
  download   Download all valid ChEMBL molecules and save full CSV
  sample-chembl  Sample N molecules from the full ChEMBL CSV
  featurize  Compute molecular features (ECFP4, FCFP4, RDKit, BRICS)
  sample     Batch-generate analogs via the PrexSyn API
  score      Score generated molecules and save summary CSV
  all        Run the full pipeline end-to-end

Usage:
  python main.py download
  python main.py sample-chembl --num-molecules 1000
  python main.py featurize
  python main.py sample
  python main.py score
  python main.py all
"""

from __future__ import annotations

import argparse
import json
import pathlib

DATA_DIR = pathlib.Path("data")

# Default file paths
CHEMBL_FULL_CSV  = DATA_DIR / "chembl_full.csv"       # all filtered molecules
CHEMBL_CSV       = DATA_DIR / "chembl_sampled.csv"    # sampled subset
CHEMBL_CACHE     = DATA_DIR / "chembl_35_chemreps.txt.gz"
FEATURES_NPZ     = DATA_DIR / "chembl_1k_features.npz"
SAMPLES_JSON     = DATA_DIR / "chembl_sampled.json"
SCORES_CSV       = DATA_DIR / "chembl_scores.csv"


# ── subcommand handlers ───────────────────────────────────────────────────────

def cmd_download(args: argparse.Namespace) -> None:
    from download_chembl import download
    download(output=args.output, cache=args.cache)


def cmd_sample_chembl(args: argparse.Namespace) -> None:
    from download_chembl import sample
    sample(input=args.input, output=args.output, num_molecules=args.num_molecules, seed=args.seed)


def cmd_featurize(args: argparse.Namespace) -> None:
    from featurize_chembl import featurize
    featurize(input=args.input, output=args.output)


def cmd_sample(args: argparse.Namespace) -> None:
    from sampler import run_batch
    run_batch(npz=args.npz, output=args.output, url=args.url,
              num_samples=args.num_samples, limit=args.limit)


def cmd_score(args: argparse.Namespace) -> None:
    from scoring import score_results, summarize

    print(f"Loading {args.input}...")
    results = json.loads(args.input.read_text())

    total_gen = sum(len(r.get("generated_smiles", [])) for r in results)
    print(f"Scoring {total_gen} generated molecules...")

    df = score_results(results,
                       similarity_threshold=args.similarity_threshold,
                       desirability_threshold=args.desirability_threshold,
                       novelty_threshold=args.novelty_threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")

    summary = summarize(df)
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    print("\n── Overall stats ──────────────────────────────")
    print(df[["tanimoto", "qed", "desirability", "is_hit", "is_similar", "is_drug_like", "is_novel"]].describe().round(3))

    print("\n── Hit rate by source molecule (top 10) ──────")
    print(summary.head(10).to_string(index=False))

    total_hits = int(df["is_hit"].sum())
    print(f"\nTotal hits: {total_hits} / {len(df)} ({100 * total_hits / len(df):.1f}%)")
    print(f"Unique scaffolds generated: {df['scaffold'].nunique()}")


def cmd_all(args: argparse.Namespace) -> None:
    # 1. Download full ChEMBL
    cmd_download(argparse.Namespace(output=CHEMBL_FULL_CSV, cache=CHEMBL_CACHE))

    # 2. Sample subset
    cmd_sample_chembl(argparse.Namespace(
        input=CHEMBL_FULL_CSV, output=CHEMBL_CSV,
        num_molecules=args.num_molecules, seed=args.seed,
    ))

    # 3. Featurize
    cmd_featurize(argparse.Namespace(input=CHEMBL_CSV, output=FEATURES_NPZ))

    # 4. Generate analogs
    cmd_sample(argparse.Namespace(
        npz=FEATURES_NPZ, output=SAMPLES_JSON,
        url=args.url, num_samples=args.num_samples, limit=None,
    ))

    # 5. Score
    cmd_score(argparse.Namespace(
        input=SAMPLES_JSON, output=SCORES_CSV,
        similarity_threshold=0.3, desirability_threshold=0.2, novelty_threshold=1.0,
    ))


# ── argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PrexSyn benchmarking pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- download --
    p_dl = sub.add_parser("download", help="Download full ChEMBL and save all valid molecules")
    p_dl.add_argument("--output", type=pathlib.Path, default=CHEMBL_FULL_CSV)
    p_dl.add_argument("--cache",  type=pathlib.Path, default=CHEMBL_CACHE)
    p_dl.set_defaults(func=cmd_download)

    # -- sample-chembl --
    p_sc = sub.add_parser("sample-chembl", help="Sample N molecules from the full ChEMBL CSV")
    p_sc.add_argument("--input",         type=pathlib.Path, default=CHEMBL_FULL_CSV)
    p_sc.add_argument("--output",        type=pathlib.Path, default=CHEMBL_CSV)
    p_sc.add_argument("--num-molecules", type=int,          default=1000)
    p_sc.add_argument("--seed",          type=int,          default=42)
    p_sc.set_defaults(func=cmd_sample_chembl)

    # -- featurize --
    p_feat = sub.add_parser("featurize", help="Compute molecular features")
    p_feat.add_argument("--input",  type=pathlib.Path, default=CHEMBL_CSV)
    p_feat.add_argument("--output", type=pathlib.Path, default=FEATURES_NPZ)
    p_feat.set_defaults(func=cmd_featurize)

    # -- sample --
    p_samp = sub.add_parser("sample", help="Batch-generate analogs via the PrexSyn API")
    p_samp.add_argument("--npz",         type=pathlib.Path, default=FEATURES_NPZ)
    p_samp.add_argument("--output",      type=pathlib.Path, default=SAMPLES_JSON)
    p_samp.add_argument("--url",         type=str,          default="http://100.65.172.100:8011/sample")
    p_samp.add_argument("--num-samples", type=int,          default=64)
    p_samp.add_argument("--limit",       type=int,          default=None,
                        help="Only process the first N molecules (for testing)")
    p_samp.set_defaults(func=cmd_sample)

    # -- score --
    p_score = sub.add_parser("score", help="Score generated molecules")
    p_score.add_argument("--input",                  type=pathlib.Path, default=SAMPLES_JSON)
    p_score.add_argument("--output",                 type=pathlib.Path, default=SCORES_CSV)
    p_score.add_argument("--similarity-threshold",   type=float,        default=0.3)
    p_score.add_argument("--desirability-threshold", type=float,        default=0.2)
    p_score.add_argument("--novelty-threshold",      type=float,        default=1.0)
    p_score.set_defaults(func=cmd_score)

    # -- all --
    p_all = sub.add_parser("all", help="Run the full pipeline end-to-end")
    p_all.add_argument("--num-molecules", type=int, default=1000)
    p_all.add_argument("--seed",          type=int, default=42)
    p_all.add_argument("--url",           type=str, default="http://100.65.172.100:8011/sample")
    p_all.add_argument("--num-samples",   type=int, default=64)
    p_all.set_defaults(func=cmd_all)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
