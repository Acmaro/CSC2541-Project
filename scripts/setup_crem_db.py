"""
Download the CReM fragment substitution database into data/crem_db/.

Downloads chembl22_sa2_hac12.db (~159 MB compressed, ~500 MB uncompressed),
the lightest pre-built database from the CReM Zenodo record 16909329.
Fragments have ≤ 12 heavy atoms, suitable for standard drug-like modifications.

Full database listing: https://zenodo.org/records/16909329

Usage:
    python scripts/setup_crem_db.py
"""

import gzip
import os
import sys
import urllib.request
from pathlib import Path

# Lightest pre-built ChEMBL22 database (fragments ≤ 12 heavy atoms, ~159 MB compressed).
# Full database listing: https://zenodo.org/records/16909329
DB_URL = "https://zenodo.org/records/16909329/files/chembl22_sa2_hac12.db.gz?download=1"
DB_NAME = "chembl22_sa2_hac12.db"
OUT_DIR = Path(__file__).parent.parent / "data" / "crem_db"
OUT_PATH = OUT_DIR / DB_NAME


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb = downloaded / 1_048_576
        total_mb = total_size / 1_048_576
        print(f"\r  {pct:5.1f}%  {mb:.0f} / {total_mb:.0f} MB", end="", flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_PATH.exists():
        print(f"Database already exists: {OUT_PATH}")
        print("Delete it manually to re-download.")
        sys.exit(0)

    gz_path = OUT_PATH.with_suffix(".db.gz")

    print("Downloading CReM database from Zenodo...")
    print(f"  URL : {DB_URL}")
    print(f"  Dest: {gz_path}")
    print("  This may take a few minutes (~159 MB compressed).")

    try:
        urllib.request.urlretrieve(DB_URL, gz_path, reporthook=_progress)
    except Exception as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        gz_path.unlink(missing_ok=True)
        sys.exit(1)

    print(f"\nDecompressing {gz_path.name} ...")
    with gzip.open(gz_path, "rb") as f_in, open(OUT_PATH, "wb") as f_out:
        f_out.write(f_in.read())
    gz_path.unlink()

    print(f"Done. Database saved to {OUT_PATH} ({OUT_PATH.stat().st_size / 1_048_576:.0f} MB)")
    print(f"\nSet the environment variable for convenience:")
    print(f"  export CREM_DB_PATH={OUT_PATH}")


if __name__ == "__main__":
    main()
