"""
build_enamine_stock.py — Convert an Enamine building-block SDF/SMILES file into
an AiZynthFinder-compatible HDF5 stock file.

Usage
-----
Download one of the following SDF files from
  https://enamine.net/building-blocks/building-blocks-catalog
and place it in data/aizynthfinder/:

  Catalog               Compounds   Recommended
  ──────────────────────────────────────────────
  Global Stock          ~825 k      ✓ (in-stock, 1–7 day delivery)
  US Stock              ~280 k        (fastest US delivery)
  Comprehensive         ~2.1 M        (too large; lowers is_solved rate)

Then run (from repo root, prexsyn conda environment):

  python scripts/build_enamine_stock.py \
      --sdf  data/aizynthfinder/enamine_global_stock.sdf \
      --out  data/aizynthfinder/enamine_stock.hdf5

Or if Enamine provides a plain SMILES file (.smi):

  python scripts/build_enamine_stock.py \
      --smi  data/aizynthfinder/enamine_global_stock.smi \
      --out  data/aizynthfinder/enamine_stock.hdf5

After running, update data/aizynthfinder/config.yml:

  stock:
    enamine: d:\\AI4DD Project\\CSC2541-Project\\data\\aizynthfinder\\enamine_stock.hdf5

Then delete any stale checkpoint files so AiZynthFinder re-scores with the new stock:

  del data\\baseline_synth_checkpoint.json
  del data\\libinvent_runs\\prexsyn_decoration\\synth_checkpoint.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sdf_to_smiles(sdf_path: Path, smi_path: Path) -> int:
    """Extract valid SMILES from an SDF file using RDKit.

    Returns the number of molecules written.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import SaltRemover
    except ImportError:
        sys.exit("[ERROR] RDKit not found. Run this script in the 'prexsyn' conda environment.")

    remover = SaltRemover.SaltRemover()
    n_written = 0
    n_skipped = 0

    print(f"Extracting SMILES from {sdf_path} ...")
    with Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True) as suppl, \
         open(smi_path, "w") as out:
        for mol in suppl:
            if mol is None:
                n_skipped += 1
                continue
            # Strip salts / counterions to get the largest fragment
            try:
                mol = remover.StripMol(mol, dontRemoveEverything=True)
            except Exception:
                pass
            smi = Chem.MolToSmiles(mol)
            if smi:
                out.write(smi + "\n")
                n_written += 1
            else:
                n_skipped += 1

            if (n_written + n_skipped) % 100_000 == 0:
                print(f"  processed {n_written + n_skipped:,} molecules "
                      f"({n_written:,} valid, {n_skipped:,} skipped) ...", flush=True)

    print(f"Done: {n_written:,} SMILES written, {n_skipped:,} skipped -> {smi_path}")
    return n_written


def run_smiles2stock(smi_path: Path, out_path: Path) -> None:
    """Call AiZynthFinder's smiles2stock CLI to build the HDF5 stock."""
    cmd = [
        sys.executable, "-m", "aizynthfinder.tools.make_stock",
        "--files", str(smi_path),
        "--output", str(out_path),
    ]
    # smiles2stock is also exposed as a console script; try that first
    console_cmd = [
        "smiles2stock",
        "--files", str(smi_path),
        "--output", str(out_path),
    ]

    print(f"\nBuilding HDF5 stock: {out_path}")
    print(f"Command: {' '.join(console_cmd)}\n")

    try:
        result = subprocess.run(console_cmd, check=True, text=True,
                                capture_output=False)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fall back to module invocation
        print("[WARN] 'smiles2stock' console script not found; trying python -m ...")
        result = subprocess.run(cmd, check=True, text=True, capture_output=False)

    print(f"\nStock file written -> {out_path}")
    print(f"Size: {out_path.stat().st_size / 1e6:.1f} MB")


def verify_stock(hdf5_path: Path, n_sample: int = 5) -> None:
    """Spot-check the generated HDF5 file using pandas."""
    try:
        import pandas as pd
    except ImportError:
        print("[WARN] pandas not available; skipping verification.")
        return

    try:
        df = pd.read_hdf(str(hdf5_path), key="table")
        print(f"\nVerification:")
        print(f"  Rows (building blocks): {len(df):,}")
        print(f"  Columns: {list(df.columns)}")
        if "inchi_key" in df.columns:
            print(f"  Sample InChI keys:")
            for key in df["inchi_key"].head(n_sample):
                print(f"    {key}")
    except Exception as e:
        print(f"[WARN] Could not verify HDF5: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an AiZynthFinder HDF5 stock from an Enamine SDF or SMILES file."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sdf", type=Path,
                        help="Path to Enamine SDF file (will be converted to SMILES first).")
    source.add_argument("--smi", type=Path,
                        help="Path to a plain SMILES file (one SMILES per line).")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output HDF5 stock file path.")
    parser.add_argument("--keep-smi", action="store_true",
                        help="Keep the intermediate SMILES file when --sdf is used.")
    args = parser.parse_args()

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.sdf:
        if not args.sdf.exists():
            sys.exit(f"[ERROR] SDF file not found: {args.sdf}")

        # Write SMILES to a sibling .smi file (or a temp file)
        smi_path = args.sdf.with_suffix(".smi")
        n = sdf_to_smiles(args.sdf, smi_path)
        if n == 0:
            sys.exit("[ERROR] No valid molecules found in SDF.")

        run_smiles2stock(smi_path, out_path)

        if not args.keep_smi:
            smi_path.unlink(missing_ok=True)
            print(f"Removed intermediate SMILES file: {smi_path}")

    else:  # --smi
        if not args.smi.exists():
            sys.exit(f"[ERROR] SMILES file not found: {args.smi}")
        run_smiles2stock(args.smi, out_path)

    verify_stock(out_path)

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Update data/aizynthfinder/config.yml:")
    print("       stock:")
    print(f"         enamine: {out_path.resolve()}")
    print("  2. Delete stale AiZynthFinder checkpoints:")
    print("       data/baseline_synth_checkpoint.json")
    print("       data/libinvent_runs/prexsyn_decoration/synth_checkpoint.json")
    print("  3. Re-run the AiZynthFinder cells in both notebooks.")
    print("=" * 60)


if __name__ == "__main__":
    main()
