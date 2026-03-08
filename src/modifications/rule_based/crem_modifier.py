"""
CReM (Context-controlled Replacement of Matched Pairs) rule-based modification module.

Wraps the `crem` library to provide fragment substitution based on a pre-built
SQLite transformation database mined from ChEMBL/ZINC.

Environment: prexsyn_env (Python 3.11)
Database:    Download replacements02_sc2.db from https://zenodo.org/record/4519690
             and set CREM_DB_PATH or pass db_path to the constructor.
"""

import os
from pathlib import Path

from crem.crem import mutate_mol  # API: mutate_mol(mol, db_name, ...)
from rdkit import Chem


class CRemModifier:
    """Wraps CReM mutate_mol to provide a simple .modify() interface.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the CReM SQLite transformation database. If None, falls back
        to the CREM_DB_PATH environment variable.
    radius : int
        Context neighbourhood radius for SMARTS matching (default 3).
        Higher values are more conservative; lower values are more permissive.
    min_size : int
        Minimum heavy-atom count of the replaced fragment (default 0).
    max_size : int
        Maximum heavy-atom count of the replaced fragment (default 10).
    min_inc : int
        Minimum change in heavy-atom count allowed (default -3).
    max_inc : int
        Maximum change in heavy-atom count allowed (default 3).
    """

    def __init__(
        self,
        db_path=None,
        radius: int = 3,
        min_size: int = 0,
        max_size: int = 10,
        min_inc: int = -3,
        max_inc: int = 3,
    ):
        resolved = db_path or os.environ.get("CREM_DB_PATH")
        if not resolved:
            raise ValueError(
                "CReM database path not provided. Pass db_path= to the constructor "
                "or set the CREM_DB_PATH environment variable.\n"
                "Download the database from: https://zenodo.org/record/4519690"
            )
        resolved = Path(resolved)
        if not resolved.exists():
            raise FileNotFoundError(
                f"CReM database not found: {resolved}\n"
                "Download replacements02_sc2.db from: https://zenodo.org/record/4519690"
            )
        self.db_path = str(resolved)
        self.radius = radius
        self.min_size = min_size
        self.max_size = max_size
        self.min_inc = min_inc
        self.max_inc = max_inc

    def modify(self, seed: str, n: int) -> list:
        """Generate up to n unique canonical SMILES variants of seed.

        Parameters
        ----------
        seed : str
            Canonical SMILES string of the input molecule.
        n : int
            Maximum number of unique variants to return.

        Returns
        -------
        list[str]
            Up to n canonical SMILES strings. Empty list if no modifications found.

        Raises
        ------
        ValueError
            If seed is not a valid SMILES string.
        """
        mol = Chem.MolFromSmiles(seed)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {seed!r}")

        results = []
        seen = set()

        for smi in mutate_mol(
            mol,
            db_name=self.db_path,
            radius=self.radius,
            min_size=self.min_size,
            max_size=self.max_size,
            min_inc=self.min_inc,
            max_inc=self.max_inc,
            return_mol=False,
        ):
            if len(results) >= n:
                break
            canon = self._canonicalize(smi)
            if canon is not None and canon not in seen:
                seen.add(canon)
                results.append(canon)

        print(f"CReM: generated {len(results)} variants from seed {seed!r}")
        return results

    @staticmethod
    def _canonicalize(smiles: str):
        """Return canonical SMILES or None if the string is invalid."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
