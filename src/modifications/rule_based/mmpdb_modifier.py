"""
mmpdb (Matched Molecular Pairs DataBase) rule-based modification module.

Wraps the `mmpdb transform` CLI command to apply learned A→B pair transformations
to a query molecule using a custom-built mmpdb transformation database.

Environment: prexsyn_env (Python 3.11)
Database:    Build with `python scripts/setup_mmpdb_db.py`
             and set MMPDB_DB_PATH or pass db_path to the constructor.
"""

import os
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem


class MmpdbModifier:
    """Wraps `mmpdb transform` CLI to provide a simple .modify() interface.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the .mmpdb SQLite file. If None, falls back to the
        MMPDB_DB_PATH environment variable.
    max_variable_heavies : int
        Maximum heavy-atom count of the swapped fragment (default 10).
        This is passed to the current mmpdb CLI as ``--max-variable-size``.
        Larger values allow more drastic changes.
    radius : int
        Minimum environment radius used for MMP context matching
        (default 3). This is passed to the current mmpdb CLI as
        ``--min-radius``. Higher values are more conservative.
    max_weight : float | None
        Reserved for future filtering support. The current mmpdb CLI
        available in this workspace does not expose a direct MW-cap flag.
    """

    def __init__(
        self,
        db_path=None,
        max_variable_heavies: int = 10,
        radius: int = 3,
        max_weight=None,
    ):
        resolved = db_path or os.environ.get("MMPDB_DB_PATH")
        if not resolved:
            raise ValueError(
                "mmpdb database path not provided. Pass db_path= to the constructor "
                "or set the MMPDB_DB_PATH environment variable.\n"
                "Build the database with: python scripts/setup_mmpdb_db.py"
            )
        resolved = Path(resolved)
        if not resolved.exists():
            raise FileNotFoundError(
                f"mmpdb database not found: {resolved}\n"
                "Build the database with: python scripts/setup_mmpdb_db.py"
            )
        self.db_path = str(resolved)
        self.max_variable_heavies = max_variable_heavies
        self.radius = radius
        self.max_weight = max_weight

    def modify(self, seed: str, n: int) -> list:
        """Generate up to n unique canonical SMILES variants of seed.

        Parameters
        ----------
        seed : str
            SMILES string of the input molecule.
        n : int
            Maximum number of unique variants to return.

        Returns
        -------
        list[str]
            Up to n canonical SMILES strings. Empty list if no modifications
            found or if mmpdb produces no output for this molecule.

        Raises
        ------
        ValueError
            If seed is not a valid SMILES string.
        RuntimeError
            If the mmpdb subprocess fails unexpectedly.
        """
        mol = Chem.MolFromSmiles(seed)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {seed!r}")

        canon_seed = Chem.MolToSmiles(mol)

        # Try with stdout first; fall back to a tempfile if not supported.
        raw_output = self._run_transform(canon_seed, output="-")
        if raw_output is None:
            raw_output = self._run_transform_tempfile(canon_seed)

        if not raw_output:
            print(f"mmpdb: generated 0 variants from seed {seed!r}")
            return []

        results = self._parse_output(raw_output, canon_seed, n)
        print(f"mmpdb: generated {len(results)} variants from seed {seed!r}")
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_cmd(self, seed_smiles: str, output: str) -> list:
        cmd = [
            "mmpdb",
            "transform",
            self.db_path,
            "--smiles", seed_smiles,
            "--max-variable-size", str(self.max_variable_heavies),
            "--min-radius", str(self.radius),
        ]
        if output != "-":
            cmd += ["--output", output]
        if self.max_weight is not None:
            raise ValueError(
                "max_weight is not supported by the installed mmpdb CLI. "
                "The current wrapper only supports the transform options "
                "available in this workspace."
            )
        return cmd

    def _run_transform(self, seed_smiles: str, output: str) -> str | None:
        """Run mmpdb transform with the given output path.

        Returns stdout text on success, None if the process failed (so the
        caller can try the tempfile fallback).
        """
        cmd = self._build_cmd(seed_smiles, output)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            # If output="-" is unsupported, return None to trigger fallback.
            if output == "-":
                return None
            raise RuntimeError(
                f"mmpdb transform failed (exit {result.returncode}):\n{result.stderr}"
            )
        return result.stdout

    def _run_transform_tempfile(self, seed_smiles: str) -> str:
        """Fallback: write mmpdb output to a named tempfile, then read it."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = self._build_cmd(seed_smiles, tmp_path)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"mmpdb transform failed (exit {result.returncode}):\n{result.stderr}"
                )
            return Path(tmp_path).read_text()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _parse_output(self, raw: str, canon_seed: str, n: int) -> list:
        """Parse TSV/CSV output from mmpdb transform into a deduplicated list."""
        lines = raw.strip().splitlines()
        if not lines:
            return []

        # Locate the SMILES column dynamically from the header.
        header = lines[0].split("\t")
        smiles_idx = None
        for i, col in enumerate(header):
            if col.strip().lower() == "smiles":
                smiles_idx = i
                break

        if smiles_idx is None:
            # Some mmpdb versions use comma-separated output.
            header = lines[0].split(",")
            for i, col in enumerate(header):
                if col.strip().lower() == "smiles":
                    smiles_idx = i
                    break

        if smiles_idx is None:
            return []

        sep = "\t" if "\t" in lines[0] else ","
        results = []
        seen = {canon_seed}

        for line in lines[1:]:
            if len(results) >= n:
                break
            parts = line.split(sep)
            if len(parts) <= smiles_idx:
                continue
            raw_smi = parts[smiles_idx].strip()
            canon = self._canonicalize(raw_smi)
            if canon is not None and canon not in seen:
                seen.add(canon)
                results.append(canon)

        return results

    @staticmethod
    def _canonicalize(smiles: str):
        """Return canonical SMILES or None if the string is invalid."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
