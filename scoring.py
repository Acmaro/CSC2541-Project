"""
Scoring functions for evaluating generated molecules from PrexSyn.

Three scoring dimensions:
  1. Tanimoto similarity  — structural similarity to the source molecule
  2. Desirability         — drug-likeness (QED, Lipinski, complexity)
  3. Hit classification   — binary label combining similarity + desirability thresholds

Usage:
    from scoring import score_results
    import json

    results = json.load(open("data/chembl_sampled.json"))
    df = score_results(results)
    print(df.describe())
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


# ── Tanimoto ──────────────────────────────────────────────────────────────────

def _ecfp4(mol: Chem.Mol):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def tanimoto(mol: Chem.Mol, ref: Chem.Mol) -> float:
    """ECFP4 Tanimoto similarity between mol and ref."""
    return float(DataStructs.TanimotoSimilarity(_ecfp4(mol), _ecfp4(ref)))


def tanimoto_to_set(mol: Chem.Mol, refs: list[Chem.Mol]) -> float:
    """Max Tanimoto similarity of mol against a set of reference molecules."""
    if not refs:
        return 0.0
    fp = _ecfp4(mol)
    return float(max(DataStructs.TanimotoSimilarity(fp, _ecfp4(r)) for r in refs))


# ── Desirability ──────────────────────────────────────────────────────────────

@dataclasses.dataclass
class DesirabilityScore:
    qed: float            # Quantitative Estimate of Drug-likeness [0, 1]
    lipinski: bool        # Passes Lipinski Rule of Five
    mw: float             # Molecular weight
    logp: float           # Wildman-Crippen LogP
    hbd: int              # H-bond donors
    hba: int              # H-bond acceptors
    tpsa: float           # Topological polar surface area
    rotatable_bonds: int  # Rotatable bond count
    score: float          # Composite desirability score [0, 1]


def desirability(mol: Chem.Mol) -> DesirabilityScore:
    """
    Compute drug-likeness desirability.

    Composite score = QED * lipinski_penalty * complexity_penalty
      - lipinski_penalty: 1.0 if passes, 0.5 if fails
      - complexity_penalty: sigmoid decay for MW > 500 and rotatable bonds > 10
    """
    qed_val   = float(QED.qed(mol))
    mw        = float(Descriptors.MolWt(mol))
    logp      = float(Descriptors.MolLogP(mol))
    hbd       = int(rdMolDescriptors.CalcNumHBD(mol))
    hba       = int(rdMolDescriptors.CalcNumHBA(mol))
    tpsa      = float(rdMolDescriptors.CalcTPSA(mol))
    rot_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(mol))

    lipinski = (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10)

    lipinski_penalty = 1.0 if lipinski else 0.5
    mw_penalty       = 1.0 / (1.0 + np.exp((mw - 500) / 50))
    rot_penalty      = 1.0 / (1.0 + np.exp((rot_bonds - 10) / 2))

    composite = qed_val * lipinski_penalty * mw_penalty * rot_penalty

    return DesirabilityScore(
        qed=qed_val,
        lipinski=lipinski,
        mw=mw,
        logp=logp,
        hbd=hbd,
        hba=hba,
        tpsa=tpsa,
        rotatable_bonds=rot_bonds,
        score=float(composite),
    )


# ── Hit classification ────────────────────────────────────────────────────────

@dataclasses.dataclass
class HitClassification:
    is_similar: bool       # Tanimoto >= similarity_threshold
    is_drug_like: bool     # desirability.score >= desirability_threshold
    is_novel: bool         # Tanimoto < novelty_threshold (not identical to source)
    is_hit: bool           # similar AND drug-like AND novel


def classify_hit(
    tanimoto_score: float,
    desirability_score: float,
    similarity_threshold: float = 0.4,
    desirability_threshold: float = 0.3,
    novelty_threshold: float = 1.0,
) -> HitClassification:
    """
    Classify a generated molecule as a hit.

    A hit must be:
      - Structurally similar to source (Tanimoto >= similarity_threshold)
      - Drug-like (desirability >= desirability_threshold)
      - Novel (Tanimoto < novelty_threshold, i.e. not an exact reconstruction)
    """
    is_similar   = tanimoto_score >= similarity_threshold
    is_drug_like = desirability_score >= desirability_threshold
    is_novel     = tanimoto_score < novelty_threshold
    return HitClassification(
        is_similar=is_similar,
        is_drug_like=is_drug_like,
        is_novel=is_novel,
        is_hit=is_similar and is_drug_like and is_novel,
    )


# ── Scaffold diversity ────────────────────────────────────────────────────────

def murcko_scaffold(mol: Chem.Mol) -> str:
    """Return the Murcko scaffold SMILES of a molecule."""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return ""


# ── Main scoring pipeline ─────────────────────────────────────────────────────

def score_molecule(
    generated_smi: str,
    source_mol: Chem.Mol,
    similarity_threshold: float = 0.3,
    desirability_threshold: float = 0.2,
    novelty_threshold: float = 1.0,
) -> dict[str, Any] | None:
    """Score a single generated SMILES against its source molecule."""
    # Discard multi-fragment outputs (salts, mixtures)
    if "." in generated_smi:
        return None

    mol = Chem.MolFromSmiles(generated_smi)
    if mol is None:
        return None

    tan  = tanimoto(mol, source_mol)
    des  = desirability(mol)
    hit  = classify_hit(tan, des.score, similarity_threshold, desirability_threshold, novelty_threshold)
    scaf = murcko_scaffold(mol)

    return {
        "generated_smiles":  generated_smi,
        "tanimoto":          round(tan, 4),
        "qed":               round(des.qed, 4),
        "desirability":      round(des.score, 4),
        "lipinski":          des.lipinski,
        "mw":                round(des.mw, 2),
        "logp":              round(des.logp, 2),
        "hbd":               des.hbd,
        "hba":               des.hba,
        "tpsa":              round(des.tpsa, 2),
        "rotatable_bonds":   des.rotatable_bonds,
        "scaffold":          scaf,
        "is_similar":        hit.is_similar,
        "is_drug_like":      hit.is_drug_like,
        "is_novel":          hit.is_novel,
        "is_hit":            hit.is_hit,
    }


def score_results(
    results: list[dict[str, Any]],
    similarity_threshold: float = 0.3,
    desirability_threshold: float = 0.2,
    novelty_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Score all generated molecules from the /sample API output.

    Args:
        results: list of dicts as returned by run_sample_batch.py
        similarity_threshold: Tanimoto >= this to be "similar"
        desirability_threshold: desirability score >= this to be "drug-like"
        novelty_threshold: Tanimoto < this to be "novel" (exclude exact reconstructions)

    Returns:
        DataFrame with one row per generated molecule.
    """
    rows: list[dict[str, Any]] = []

    for entry in results:
        source_smi = entry.get("source_smiles")
        if not source_smi:
            continue
        source_mol = Chem.MolFromSmiles(source_smi)
        if source_mol is None:
            continue

        for gen_smi in entry.get("generated_smiles", []):
            row = score_molecule(
                gen_smi, source_mol,
                similarity_threshold, desirability_threshold, novelty_threshold,
            )
            if row is None:
                continue
            row["source_smiles"] = source_smi
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Reorder columns
    front = ["source_smiles", "generated_smiles", "tanimoto", "qed",
             "desirability", "is_hit", "is_similar", "is_drug_like", "is_novel"]
    rest  = [c for c in df.columns if c not in front]
    return df[front + rest]


# ── Summary stats ─────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-source-molecule summary statistics."""
    return (
        df.groupby("source_smiles")
        .agg(
            n_generated    =("generated_smiles", "count"),
            n_hits         =("is_hit", "sum"),
            hit_rate       =("is_hit", "mean"),
            mean_tanimoto  =("tanimoto", "mean"),
            max_tanimoto   =("tanimoto", "max"),
            mean_qed       =("qed", "mean"),
            mean_desirability=("desirability", "mean"),
            n_unique_scaffolds=("scaffold", "nunique"),
        )
        .reset_index()
        .sort_values("hit_rate", ascending=False)
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score generated molecules")
    parser.add_argument("--input",  type=pathlib.Path, default=pathlib.Path("data/chembl_sampled.json"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data/chembl_scores.csv"))
    parser.add_argument("--similarity-threshold",   type=float, default=0.3)
    parser.add_argument("--desirability-threshold", type=float, default=0.2)
    parser.add_argument("--novelty-threshold",      type=float, default=1.0)
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    results = json.loads(args.input.read_text())

    print(f"Scoring {sum(len(r['generated_smiles']) for r in results)} generated molecules...")
    df = score_results(
        results,
        similarity_threshold=args.similarity_threshold,
        desirability_threshold=args.desirability_threshold,
        novelty_threshold=args.novelty_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")

    print("\n── Overall stats ──────────────────────────────")
    print(df[["tanimoto", "qed", "desirability", "is_hit", "is_similar", "is_drug_like", "is_novel"]].describe().round(3))

    print("\n── Hit rate by source molecule (top 10) ──────")
    print(summarize(df).head(10).to_string(index=False))

    total_hits = df["is_hit"].sum()
    print(f"\nTotal hits: {total_hits} / {len(df)} ({100*total_hits/len(df):.1f}%)")
    print(f"Unique scaffolds generated: {df['scaffold'].nunique()}")
