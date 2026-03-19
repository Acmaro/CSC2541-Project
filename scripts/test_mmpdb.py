"""
Smoke test for MmpdbModifier.

Usage:
    python scripts/test_mmpdb.py [--db PATH]

If --db is omitted, reads MMPDB_DB_PATH from the environment.
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modifications.rule_based.mmpdb_modifier import MmpdbModifier

# A known ChEMBL drug-like molecule (sildenafil scaffold)
SEED = "CCc1nn(C)c2ccc(cc12)C(=O)NCc3ccccc3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        default=None,
        help="Path to .mmpdb database (defaults to data/mmpdb_db/chembl_50k.mmpdb)",
    )
    args = parser.parse_args()

    db_path = args.db or Path(__file__).parent.parent / "data" / "mmpdb_db" / "chembl_50k.mmpdb"

    print(f"Database : {db_path}")
    print(f"Seed     : {SEED}")
    print(f"N        : 5")
    print()

    M = MmpdbModifier(db_path=str(db_path))
    variants = M.modify(SEED, n=5)

    print(f"\nVariants returned: {len(variants)}")
    for i, smi in enumerate(variants, 1):
        print(f"  {i}. {smi}")

    # Basic assertions
    assert isinstance(variants, list), "Expected list"
    assert all(isinstance(s, str) for s in variants), "Expected list of strings"
    assert SEED not in variants, "Seed should be filtered from output"

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
