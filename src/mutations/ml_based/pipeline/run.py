"""
run.py — End-to-end Lib-INVENT fingerprint-mutation pipeline.

Usage (from the project root, with lib-invent conda env active):

    python pipeline/run.py \
        --input  "CCc1nn(C)c2ccc(cc12)C(=O)NCc3ccccc3" \
        --target "CCc1nn(C)c2ccc(cc12)C(=O)NCCN3CCOCC3" \
        --run-dir runs/my_run

Optional flags:
    --method   brics|recap|both   (fragmentation, default: brics)
    --n-steps  INT                (RL steps, default: 200)
    --batch    INT                (batch size, default: 64)
    --top      INT                (top-N results to print, default: 20)
    --qed      FLOAT              (QED objective weight, 0 = off, default: 0)
    --model    PATH               (pretrained model; default: reaction_based.model)
    --skip-rl                     (skip RL, run scaffold_decorating directly)

Outputs (inside --run-dir):
    scaffolds.smi         — extracted scaffolds fed to RL
    rl_config.json        — generated RL config
    decorate_config.json  — generated scaffold_decorating config
    rl_model.pt           — fine-tuned model (after RL)
    decorated.csv         — final generated SMILES
    logs/                 — Lib-INVENT logs
"""

import argparse
import csv
import os
import subprocess
import sys

from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem

from fragment import get_scaffolds
from configs import make_rl_config, make_decorate_config, write_json, write_scaffolds_smi

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LIB_INVENT_DIR = os.path.join(PROJECT_ROOT, "Lib-INVENT")
INPUT_PY = os.path.join(LIB_INVENT_DIR, "input.py")
DEFAULT_MODEL = os.path.join(LIB_INVENT_DIR, "trained_models", "reaction_based.model")


# ---------------------------------------------------------------------------
# Target parser — accepts plain SMILES or ECFP4(smiles) notation
# ---------------------------------------------------------------------------

import re as _re

_FP_PATTERN = _re.compile(r"^ECFP4\((.+)\)$", _re.IGNORECASE)


def _parse_target(target_arg: str) -> str:
    """
    Accept either:
      - plain SMILES: "CCc1nn(C)c2ccc(...)"
      - fingerprint notation: "ECFP4(CCc1nn(C)c2ccc(...))"

    In both cases returns the inner SMILES string.
    The tanimoto_similarity scoring component computes ECFP4 internally,
    so ECFP4(X) and X are equivalent for RL scoring and ranking.
    """
    m = _FP_PATTERN.match(target_arg.strip())
    if m:
        return m.group(1).strip()
    return target_arg.strip()


# ---------------------------------------------------------------------------
# Similarity helper (for ranking final output)
# ---------------------------------------------------------------------------

def _ecfp4(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def _tanimoto(fp1, fp2) -> float:
    if fp1 is None or fp2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def _run_lib_invent(config_path: str) -> None:
    """Call `python input.py <config>` from inside Lib-INVENT/."""
    cmd = [sys.executable, INPUT_PY, config_path]
    print(f"\n[pipeline] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=LIB_INVENT_DIR)
    if result.returncode != 0:
        sys.exit(f"[pipeline] Lib-INVENT exited with code {result.returncode}")


# ---------------------------------------------------------------------------
# Results reader
# ---------------------------------------------------------------------------

def _read_decorated_csv(csv_path: str) -> list[dict]:
    """
    Parse Lib-INVENT scaffold_decorating output CSV.
    Expected columns: Scaffold, Decoration, SMILES (full molecule), NLL
    Returns list of dicts.
    """
    rows = []
    if not os.path.exists(csv_path):
        print(f"[pipeline] Warning: output file not found: {csv_path}")
        return rows
    with open(csv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def _smiles_column(rows: list[dict]) -> str:
    """Detect which column holds the full assembled SMILES."""
    if not rows:
        return "SMILES"
    for candidate in ("SMILES", "Smiles", "smiles", "output_smiles", "molecules"):
        if candidate in rows[0]:
            return candidate
    # Fall back to first column that looks like SMILES
    for col in rows[0]:
        if any(c in rows[0][col] for c in ("C", "c", "N", "O")):
            return col
    return list(rows[0].keys())[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Lib-INVENT fingerprint-mutation pipeline")
    parser.add_argument("--input",   required=True, help="Input SMILES to mutate")
    parser.add_argument("--target",  required=True, help="Target SMILES (fingerprint reference)")
    parser.add_argument("--run-dir", default="runs/default", help="Output directory")
    parser.add_argument("--method",  default="brics", choices=["brics", "recap", "both"])
    parser.add_argument("--n-steps", type=int,   default=200)
    parser.add_argument("--batch",   type=int,   default=64)
    parser.add_argument("--top",     type=int,   default=20)
    parser.add_argument("--qed",     type=float, default=0.0)
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--skip-rl", action="store_true",
                        help="Skip RL fine-tuning and decorate with the pretrained model")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    target_smiles = _parse_target(args.target)
    if target_smiles != args.target:
        print(f"[pipeline] Target parsed from ECFP4(...): {target_smiles}")

    model_abs = os.path.abspath(args.model)
    if not os.path.exists(model_abs):
        sys.exit(f"[pipeline] Model not found: {model_abs}")

    # ── Step 1: Fragment ─────────────────────────────────────────────────────
    print(f"\n[pipeline] Step 1: Fragmenting '{args.input}' ({args.method})")
    scaffolds = get_scaffolds(args.input, method=args.method)

    if not scaffolds:
        sys.exit(
            "[pipeline] No valid scaffolds found. "
            "Try a larger molecule or --method recap."
        )

    print(f"[pipeline] {len(scaffolds)} scaffold(s):")
    for s in scaffolds:
        print(f"  {s}")

    scaffolds_path = os.path.join(run_dir, "scaffolds.smi")
    write_scaffolds_smi(scaffolds, scaffolds_path)

    # ── Step 2: RL fine-tuning ───────────────────────────────────────────────
    rl_model_path = os.path.join(run_dir, "rl_model.pt")

    if not args.skip_rl:
        print(f"\n[pipeline] Step 2: RL fine-tuning ({args.n_steps} steps)")

        rl_config = make_rl_config(
            scaffolds=scaffolds,
            target_smiles=target_smiles,
            actor_path=model_abs,
            critic_path=model_abs,
            output_model_path=rl_model_path,
            logging_path=os.path.join(run_dir, "logs", "rl"),
            n_steps=args.n_steps,
            batch_size=args.batch,
            qed_weight=args.qed,
        )
        rl_config_path = os.path.join(run_dir, "rl_config.json")
        write_json(rl_config, rl_config_path)
        print(f"[pipeline] RL config written to {rl_config_path}")

        _run_lib_invent(rl_config_path)

        if not os.path.exists(rl_model_path):
            print(
                "[pipeline] Warning: expected RL output model not found at "
                f"{rl_model_path}. Falling back to pretrained model for decoration."
            )
            rl_model_path = model_abs
    else:
        print("[pipeline] Step 2: Skipping RL (--skip-rl)")
        rl_model_path = model_abs

    # ── Step 3: Scaffold decorating (extra sampling with pretrained model) ───
    print("\n[pipeline] Step 3: Generating additional decorated molecules")

    output_csv = os.path.join(run_dir, "decorated.csv")
    decorate_config = make_decorate_config(
        model_path=model_abs,          # always use pretrained for decoration
        scaffolds_smi_path=scaffolds_path,
        output_csv_path=output_csv,
        logging_path=os.path.join(run_dir, "logs", "decorate"),
        n_decorations=128,
    )
    decorate_config_path = os.path.join(run_dir, "decorate_config.json")
    write_json(decorate_config, decorate_config_path)

    result = subprocess.run(
        [sys.executable, INPUT_PY, decorate_config_path],
        cwd=LIB_INVENT_DIR,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[pipeline] scaffold_decorating failed (non-fatal): {result.stderr[-300:]}")

    # ── Step 4: Rank and display results ────────────────────────────────────
    # Primary source: scaffold_memory.csv written by the RL diversity filter
    # (contains all unique SMILES seen during training, scored).
    # Fallback: decorated.csv from scaffold_decorating.
    print("\n[pipeline] Step 4: Ranking results by Tanimoto similarity to target")

    scaffold_memory_csv = os.path.join(run_dir, "logs", "rl", "scaffold_memory.csv")
    target_fp = _ecfp4(target_smiles)
    ranked = []
    seen: set[str] = set()

    def _collect(csv_path: str, smiles_col: str, sep: str = ",") -> None:
        if not os.path.exists(csv_path):
            return
        with open(csv_path) as f:
            reader = csv.DictReader(f, delimiter=sep)
            for row in reader:
                smi = row.get(smiles_col, "").strip()
                if not smi or smi == "INVALID" or smi in seen:
                    continue
                seen.add(smi)
                sim = _tanimoto(_ecfp4(smi), target_fp)
                ranked.append((sim, smi))

    _collect(scaffold_memory_csv, "SMILES", sep=",")

    if os.path.exists(output_csv):
        rows = _read_decorated_csv(output_csv)
        if rows:
            _collect(output_csv, _smiles_column(rows), sep="\t")

    if not ranked:
        print("[pipeline] No output molecules found.")
        return

    ranked.sort(reverse=True)
    print(f"[pipeline] Total unique molecules: {len(ranked)}")
    print(f"\n{'Rank':<5} {'Tanimoto':>8}  SMILES")
    print("-" * 80)
    for rank, (sim, smi) in enumerate(ranked[: args.top], 1):
        print(f"{rank:<5} {sim:>8.4f}  {smi}")

    print(f"\n[pipeline] Done. Full RL output: {scaffold_memory_csv}")


if __name__ == "__main__":
    main()
