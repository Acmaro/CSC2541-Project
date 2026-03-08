"""
Smoke test for CRemModifier.

Usage:
    python scripts/test_crem.py [--db PATH]

If --db is omitted, reads CREM_DB_PATH from the environment.
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modifications.rule_based.crem_modifier import CRemModifier

# A known ChEMBL drug-like molecule (sildenafil scaffold)
SEED = "CCc1nn(C)c2ccc(cc12)C(=O)NCc3ccccc3"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None, help="Path to CReM SQLite database")
    args = parser.parse_args()

    db_path = args.db or Path(__file__).parent.parent / "data" / "crem_db" / "chembl22_sa2_hac12.db"

    print(f"Database : {db_path}")
    print(f"Seed     : {SEED}")
    print(f"N        : 5")
    print()

    M = CRemModifier(db_path=str(db_path))
    variants = M.modify(SEED, n=5)

    print(f"\nVariants returned: {len(variants)}")
    for i, smi in enumerate(variants, 1):
        print(f"  {i}. {smi}")

    # Basic assertions
    assert isinstance(variants, list), "Expected list"
    assert all(isinstance(s, str) for s in variants), "Expected list of strings"

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
