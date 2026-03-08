"""
fragment.py — Decompose an input SMILES into Lib-INVENT scaffolds.

Uses BRICS (primary) with a RECAP fallback.  Only fragments with 1–3
attachment points are kept as candidate scaffolds; tiny pieces (< 5 heavy
atoms) are discarded.

Attachment points are renumbered to [*:0], [*:1], … as Lib-INVENT expects.
"""

import re
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Recap import RecapDecompose


# ── helpers ──────────────────────────────────────────────────────────────────

_DUMMY_PAT = re.compile(r"\[\d+\*\]|\[\*\]|\[\*:\d+\]")


def _count_attachment_points(smiles: str) -> int:
    return len(_DUMMY_PAT.findall(smiles))


def _renumber_attachment_points(smiles: str) -> str:
    """Renumber dummy atoms sequentially to [*:0], [*:1], …"""
    counter = [0]

    def _replace(_match):
        label = f"[*:{counter[0]}]"
        counter[0] += 1
        return label

    return _DUMMY_PAT.sub(_replace, smiles)


def _heavy_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def _is_valid_scaffold(smiles: str, min_heavy: int = 5, max_attach: int = 3) -> bool:
    n = _count_attachment_points(smiles)
    return 1 <= n <= max_attach and _heavy_atom_count(smiles) >= min_heavy


# ── BRICS ─────────────────────────────────────────────────────────────────────

def brics_scaffolds(smiles: str) -> list[str]:
    """
    Return BRICS fragments that qualify as scaffolds (have attachment points
    and are large enough).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    frags = BRICS.BRICSDecompose(mol, returnMols=False)
    candidates = []
    for f in frags:
        relabeled = _renumber_attachment_points(f)
        if _is_valid_scaffold(relabeled):
            candidates.append(relabeled)
    # Deduplicate by canonical SMILES (ignore attachment point numbers)
    seen, unique = set(), []
    for f in candidates:
        key = re.sub(r"\[\*:\d+\]", "[*]", f)
        mol_check = Chem.MolFromSmiles(key)
        key = Chem.MolToSmiles(mol_check) if mol_check else key
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


# ── RECAP ─────────────────────────────────────────────────────────────────────

def recap_scaffolds(smiles: str) -> list[str]:
    """
    Return RECAP fragments (internal nodes, which are the scaffold pieces
    with attachment points) as candidate scaffolds.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    tree = RecapDecompose(mol)
    # Include all non-root, non-leaf nodes (the scaffold "cores")
    candidates = []
    # allNodes includes root + internal + leaves
    for node in tree.GetAllChildren().values():
        f = _renumber_attachment_points(node.smiles)
        if _is_valid_scaffold(f):
            candidates.append(f)
    return candidates


# ── public API ────────────────────────────────────────────────────────────────

def get_scaffolds(smiles: str, method: str = "brics") -> list[str]:
    """
    Main entry point.  Returns a list of scaffold SMILES with attachment
    points labelled [*:0], [*:1], …

    method: "brics" | "recap" | "both"
    """
    scaffolds: list[str] = []

    if method in ("brics", "both"):
        scaffolds.extend(brics_scaffolds(smiles))

    if method in ("recap", "both") or (method == "brics" and not scaffolds):
        scaffolds.extend(recap_scaffolds(smiles))

    if not scaffolds:
        # Last resort: treat the whole molecule as a single scaffold with one
        # attachment point so the pipeline can still run (low utility but
        # allows testing the rest of the pipeline).
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            print(
                f"[fragment] Warning: no fragments found for '{smiles}'. "
                "Returning original molecule as a placeholder scaffold."
            )

    return scaffolds


if __name__ == "__main__":
    import sys

    smi = sys.argv[1] if len(sys.argv) > 1 else "CCc1nn(C)c2ccc(cc12)C(=O)NCc3ccccc3"
    result = get_scaffolds(smi)
    print(f"Input : {smi}")
    print(f"Scaffolds ({len(result)}):")
    for s in result:
        print(f"  {s}  [{_count_attachment_points(s)} attach pts, "
              f"{_heavy_atom_count(s)} heavy atoms]")
