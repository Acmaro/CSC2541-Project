"""
Build the mmpdb transformation database into data/mmpdb_db/.

Steps:
  1. Download/cache full ChEMBL SMILES corpus (via src.utils.download_chembl).
  2. Sample N molecules and write a plain .smi file (one SMILES per line).
  3. Run `mmpdb fragment` to generate fragment pairs (~5-20 min for 50k).
  4. Run `mmpdb index` to build the SQLite .mmpdb transformation database.

Usage:
    python scripts/setup_mmpdb_db.py [--n-mols 50000] [--num-jobs 4]
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.download_chembl import download, sample

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
CHEMBL_CSV = DATA_DIR / "chembl_filtered.csv"
OUT_DIR = DATA_DIR / "mmpdb_db"
SMI_FILE = OUT_DIR / "chembl_50k.smi"
FRAG_FILE = OUT_DIR / "chembl_50k.fragments"
DB_FILE = OUT_DIR / "chembl_50k.mmpdb"


def _run(cmd: list, desc: str) -> None:
    print(f"\n[step] {desc}")
    print("  $", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nCommand failed (exit {result.returncode}). Aborting.", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Build mmpdb transformation database.")
    parser.add_argument(
        "--n-mols",
        type=int,
        default=50_000,
        help="Number of molecules to sample from ChEMBL (default: 50000)",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=4,
        help="Parallel jobs for mmpdb fragment (default: 4)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Idempotency check.
    if DB_FILE.exists():
        print(f"Database already exists: {DB_FILE}")
        print("Delete it manually to rebuild.")
        sys.exit(0)

    # Step 1 — ensure full ChEMBL CSV is present.
    if not CHEMBL_CSV.exists():
        print(f"ChEMBL filtered CSV not found at {CHEMBL_CSV}. Downloading now...")
        download(output=CHEMBL_CSV, cache=DATA_DIR / "chembl_raw.gz")
    else:
        print(f"Using cached ChEMBL CSV: {CHEMBL_CSV}")

    # Step 2 — sample N molecules → plain .smi file.
    if not SMI_FILE.exists():
        tmp_csv = OUT_DIR / "chembl_sample.csv"
        sample(input=CHEMBL_CSV, output=tmp_csv, num_molecules=args.n_mols)

        import pandas as pd
        df = pd.read_csv(tmp_csv)
        smiles_list = df["SMILES"].dropna().tolist()
        SMI_FILE.write_text("\n".join(smiles_list) + "\n")
        tmp_csv.unlink(missing_ok=True)
        print(f"Wrote {len(smiles_list):,} SMILES to {SMI_FILE}")
    else:
        print(f"Reusing existing SMILES file: {SMI_FILE}")

    # Step 3 — fragment.
    if not FRAG_FILE.exists():
        _run(
            [
                "mmpdb", "fragment",
                str(SMI_FILE),
                "--output", str(FRAG_FILE),
                "--num-jobs", str(args.num_jobs),
            ],
            desc=f"mmpdb fragment  (this may take 5-20 min for {args.n_mols:,} molecules)",
        )
    else:
        print(f"Reusing existing fragments file: {FRAG_FILE}")

    # Step 4 — index.
    _run(
        [
            "mmpdb", "index",
            str(FRAG_FILE),
            "--output", str(DB_FILE),
            "--title", "ChEMBL 50k MMP index",
        ],
        desc="mmpdb index",
    )

    size_mb = DB_FILE.stat().st_size / 1_048_576
    print(f"\nDone. Database saved to {DB_FILE} ({size_mb:.0f} MB)")
    print("\nSet the environment variable for convenience:")
    print(f"  export MMPDB_DB_PATH={DB_FILE.resolve()}")


if __name__ == "__main__":
    main()
