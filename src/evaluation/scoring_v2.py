"""
scoring_v2.py — Spec-relative evaluation pipeline for post-generation molecular modification.

Implements the full evaluation design from paper-draft.md §2.3:

  Gate 1 — Synthesizability (caller responsibility: pass only synthesizable variants)
  Gate 2 — Property conservation:
      Substructural:     ECFP4 Tanimoto vs spec fingerprint  (tau_t)
      Physicochemical:   spec-relative desirability, geometric mean of 6 descriptor deltas (tau_d)
  Supplementary — ESPsim 3D shape + electrostatic similarity (requires espsim package)

  Aggregation:
      hit rate, unique hits, expansion factor vs baseline
      chemical diversity (1 - mean pairwise Tanimoto among hits)
      stratification by baseline quality bin (<0.5, 0.5-0.7, 0.7-0.85, 0.85-1.0)
      complementarity: pairwise Jaccard overlap between method hit sets

Usage:
    from scoring_v2 import make_spec, score_batch, summarize, complementarity, best_combinations

    spec = make_spec("CC(=O)Oc1ccccc1C(=O)O")       # ChEMBL reference
    df   = score_batch(variant_smiles, spec, baseline_quality=0.72, method="CReM")
    summary = summarize(df)
    print(summary)

Input JSON schema for CLI (--input):
    [
      {
        "spec_smiles":       "<ChEMBL SMILES>",
        "baseline_quality":  0.72,
        "methods": {
          "CReM":     ["<SMILES>", ...],
          "mmpdb":    ["<SMILES>", ...],
          "baseline": ["<SMILES>", ...]
        }
      },
      ...
    ]
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import time
from itertools import combinations
from typing import Any, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

try:
    from espsim import GetEspSim, GetShapeSim
    _ESPSIM_AVAILABLE = True
except ImportError:
    _ESPSIM_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

# Per-descriptor tolerances (from paper-draft.md pseudocode)
TOLERANCES: dict[str, float] = {
    "mw":        50.0,   # ±50 Da
    "clogp":      1.0,   # ±1.0
    "tpsa":      20.0,   # ±20 Å²
    "hbd":        1.0,   # ±1
    "hba":        2.0,   # ±2
    "rot_bonds":  2.0,   # ±2
}

# Baseline quality bins (Tanimoto of PrexSyn seed to spec)
BINS       = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
BIN_LABELS = ["<0.5", "0.5-0.7", "0.7-0.85", "0.85-1.0"]

# Default multi-threshold values (paper §2.3: "results at multiple threshold values")
DEFAULT_TAU_T_LIST: list[float] = [0.6, 0.7, 0.85]
DEFAULT_TAU_D: float = 0.8


# ── MolSpec ───────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class MolSpec:
    """
    Property specification derived from a ChEMBL reference molecule.

    The reference molecule is NOT an optimization target. It provides:
      - ECFP4 fingerprint for substructural conservation scoring
      - Descriptor values for physicochemical conservation scoring
      - Optional 3D conformer for ESPsim (supplementary)
    """
    spec_smiles: str
    fp:          DataStructs.ExplicitBitVect  # ECFP4 bitvector (radius=2, 2048 bits)
    mw:          float
    clogp:       float
    tpsa:        float
    hbd:         int
    hba:         int
    rot_bonds:   int
    mol3d:       Optional[Chem.Mol] = None   # 3D conformer for ESPsim


def make_spec(smiles: str, generate_conformer: bool = False) -> MolSpec | None:
    """
    Build a MolSpec from a SMILES string.

    Args:
        smiles:             SMILES of the ChEMBL reference molecule.
        generate_conformer: If True, generate a 3D ETKDG conformer for ESPsim.
                            Adds ~0.5-2 s per molecule; set False unless ESPsim is needed.

    Returns:
        MolSpec or None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp        = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    mw        = float(Descriptors.MolWt(mol))
    clogp     = float(Descriptors.MolLogP(mol))
    tpsa      = float(rdMolDescriptors.CalcTPSA(mol))
    hbd       = int(rdMolDescriptors.CalcNumHBD(mol))
    hba       = int(rdMolDescriptors.CalcNumHBA(mol))
    rot_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    mol3d     = _generate_conformer(mol) if generate_conformer else None

    return MolSpec(
        spec_smiles=smiles,
        fp=fp,
        mw=mw, clogp=clogp, tpsa=tpsa,
        hbd=hbd, hba=hba, rot_bonds=rot_bonds,
        mol3d=mol3d,
    )


# ── Fingerprint & Tanimoto ────────────────────────────────────────────────────

def _ecfp4(mol: Chem.Mol) -> DataStructs.ExplicitBitVect:
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def tanimoto_to_spec(mol: Chem.Mol, spec: MolSpec) -> float:
    """ECFP4 Tanimoto similarity between a variant and the spec fingerprint."""
    return float(DataStructs.TanimotoSimilarity(_ecfp4(mol), spec.fp))


# ── Desirability ──────────────────────────────────────────────────────────────

def _linear_desirability(value: float, target: float, tolerance: float) -> float:
    """
    Linear tolerance function mapping descriptor delta to [0, 1].
    Returns 1.0 when value == target, 0.0 when |value - target| >= tolerance.
    """
    return max(0.0, 1.0 - abs(value - target) / tolerance)


def _geometric_mean(scores: list[float]) -> float:
    """
    Geometric mean of a list of [0, 1] scores.
    Returns 0.0 if any score is zero (one bad descriptor kills the aggregate).
    """
    if any(s <= 0.0 for s in scores):
        return 0.0
    return float(math.exp(sum(math.log(s) for s in scores) / len(scores)))


@dataclasses.dataclass
class DesirabilityResult:
    mw_score:        float  # MW conservation component
    clogp_score:     float  # CLogP conservation component
    tpsa_score:      float  # TPSA conservation component
    hbd_score:       float  # HBD conservation component
    hba_score:       float  # HBA conservation component
    rot_bonds_score: float  # Rotatable bonds conservation component
    score:           float  # Geometric mean of all components [0, 1]


def desirability(mol: Chem.Mol, spec: MolSpec) -> DesirabilityResult:
    """
    Spec-relative physicochemical conservation score.

    Each of the 6 descriptors is compared to the corresponding spec target via a
    linear tolerance function (TOLERANCES dict). The 6 component scores are then
    aggregated by geometric mean: a single violated descriptor drives the score
    toward zero, matching the paper's design intent.
    """
    mw_s    = _linear_desirability(Descriptors.MolWt(mol),                          spec.mw,        TOLERANCES["mw"])
    clogp_s = _linear_desirability(Descriptors.MolLogP(mol),                        spec.clogp,     TOLERANCES["clogp"])
    tpsa_s  = _linear_desirability(rdMolDescriptors.CalcTPSA(mol),                  spec.tpsa,      TOLERANCES["tpsa"])
    hbd_s   = _linear_desirability(rdMolDescriptors.CalcNumHBD(mol),                spec.hbd,       TOLERANCES["hbd"])
    hba_s   = _linear_desirability(rdMolDescriptors.CalcNumHBA(mol),                spec.hba,       TOLERANCES["hba"])
    rot_s   = _linear_desirability(rdMolDescriptors.CalcNumRotatableBonds(mol),     spec.rot_bonds, TOLERANCES["rot_bonds"])

    score = _geometric_mean([mw_s, clogp_s, tpsa_s, hbd_s, hba_s, rot_s])

    return DesirabilityResult(
        mw_score=mw_s, clogp_score=clogp_s, tpsa_score=tpsa_s,
        hbd_score=hbd_s, hba_score=hba_s, rot_bonds_score=rot_s,
        score=score,
    )


# ── ESPsim (supplementary) ────────────────────────────────────────────────────

def _generate_conformer(mol: Chem.Mol) -> Chem.Mol | None:
    """
    Generate a 3D ETKDG conformer with MMFF optimization.
    Returns None if embedding fails (e.g., macrocycles, disconnected fragments).
    """
    mol3d = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol3d, params) == -1:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol3d)
    except Exception:
        pass
    return mol3d


def espsim_score(mol: Chem.Mol, spec: MolSpec) -> float | None:
    """
    Combined 3D shape + electrostatic similarity (ESPsim), averaged equally.

    Returns None if:
      - espsim package is not installed
      - spec.mol3d is None (make_spec called without generate_conformer=True)
      - 3D conformer generation fails for the variant

    Note: this metric couples evaluation to the specific 3D geometry of the
    ChEMBL reference conformer. Report as supplementary diagnostic only.
    """
    if not _ESPSIM_AVAILABLE or spec.mol3d is None:
        return None
    mol3d = _generate_conformer(mol)
    if mol3d is None:
        return None
    try:
        shape = float(GetShapeSim(mol3d, spec.mol3d))
        esp   = float(GetEspSim(mol3d, spec.mol3d))
        return (shape + esp) / 2.0
    except Exception:
        return None


# ── Single-variant scoring ────────────────────────────────────────────────────

def _assign_bin(quality: float) -> str:
    for (lo, hi), label in zip(BINS, BIN_LABELS):
        if lo <= quality < hi:
            return label
    return BIN_LABELS[-1]


def score_variant(
    variant_smi:      str,
    spec:             MolSpec,
    baseline_quality: float,
    compute_espsim:   bool = False,
) -> dict[str, Any] | None:
    """
    Score a single variant SMILES against a property specification.

    Synthesizability (Gate 1) is the caller's responsibility: pass only variants
    that have already passed AiZynthFinder retrosynthetic analysis.

    Args:
        variant_smi:       SMILES of the variant molecule.
        spec:              MolSpec derived from the ChEMBL reference.
        baseline_quality:  Tanimoto(PrexSyn seed, spec.fp) — determines quality bin.
        compute_espsim:    Compute ESPsim supplementary metric (slow).

    Returns:
        Dict of scores, or None for multi-fragment or invalid SMILES.
    """
    if "." in variant_smi:
        return None
    mol = Chem.MolFromSmiles(variant_smi)
    if mol is None:
        return None

    tan = tanimoto_to_spec(mol, spec)
    des = desirability(mol, spec)
    esp = espsim_score(mol, spec) if compute_espsim else None

    return {
        "variant_smiles":    variant_smi,
        "spec_smiles":       spec.spec_smiles,
        "tanimoto":          round(tan, 4),
        "desirability":      round(des.score, 4),
        "mw_score":          round(des.mw_score, 4),
        "clogp_score":       round(des.clogp_score, 4),
        "tpsa_score":        round(des.tpsa_score, 4),
        "hbd_score":         round(des.hbd_score, 4),
        "hba_score":         round(des.hba_score, 4),
        "rot_bonds_score":   round(des.rot_bonds_score, 4),
        "espsim":            round(esp, 4) if esp is not None else None,
        "baseline_quality":  round(baseline_quality, 4),
        "quality_bin":       _assign_bin(baseline_quality),
    }


# ── Batch scoring ─────────────────────────────────────────────────────────────

def score_batch(
    variants:         list[str],
    spec:             MolSpec,
    baseline_quality: float,
    method:           str,
    compute_espsim:   bool = False,
) -> pd.DataFrame:
    """
    Score a list of variant SMILES for one method against one spec.

    Args:
        variants:         List of variant SMILES (already deduplicated by caller).
        spec:             MolSpec for this specification.
        baseline_quality: Tanimoto(PrexSyn seed, spec) for bin assignment.
        method:           Method label, e.g. "CReM", "mmpdb", "baseline".
        compute_espsim:   Compute ESPsim supplementary metric.

    Returns:
        DataFrame with one row per valid variant, plus a "method" column.
    """
    rows = []
    for smi in variants:
        row = score_variant(smi, spec, baseline_quality, compute_espsim)
        if row is not None:
            row["method"] = method
            rows.append(row)
    return pd.DataFrame(rows)


# ── Hit classification ────────────────────────────────────────────────────────

def classify_hits(df: pd.DataFrame, tau_t: float, tau_d: float) -> pd.Series:
    """
    Boolean Series: True where tanimoto >= tau_t AND desirability >= tau_d.
    A hit must conserve both substructural and physicochemical properties.
    """
    return (df["tanimoto"] >= tau_t) & (df["desirability"] >= tau_d)


# ── Expansion factor ──────────────────────────────────────────────────────────

def expansion_factor(method_hits: set[str], baseline_hits: set[str]) -> float:
    """
    Unique method hits / unique baseline hits.

    The core measure of marginal value: a value > 1 means the method produces
    synthesizable, property-conserving candidates that repeated PrexSyn sampling
    cannot reach. Returns inf if the baseline has zero hits.
    """
    if len(baseline_hits) == 0:
        return float("inf")
    return len(method_hits) / len(baseline_hits)


# ── Complementarity ───────────────────────────────────────────────────────────

def complementarity(hit_sets: dict[str, set[str]]) -> pd.DataFrame:
    """
    Pairwise Jaccard overlap matrix between method hit sets.

    Low values indicate complementary methods (non-overlapping hits);
    high values indicate redundancy.

    Args:
        hit_sets: dict mapping method name to set of hit SMILES strings.

    Returns:
        DataFrame[method x method] with Jaccard coefficients in [0, 1].
    """
    methods = list(hit_sets.keys())
    matrix: dict[str, dict[str, float]] = {}
    for m1 in methods:
        matrix[m1] = {}
        for m2 in methods:
            union = hit_sets[m1] | hit_sets[m2]
            inter = hit_sets[m1] & hit_sets[m2]
            matrix[m1][m2] = len(inter) / len(union) if union else 0.0
    return pd.DataFrame(matrix, index=methods)


def best_combinations(
    hit_sets: dict[str, set[str]],
    top_k:    int = 5,
) -> list[tuple[tuple[str, ...], int]]:
    """
    Rank all 2- and 3-method combinations by total unique hit coverage.

    The "baseline" key is excluded from combination search.

    Returns:
        List of (method_tuple, n_unique_hits) sorted by coverage descending.
    """
    methods = [m for m in hit_sets if m != "baseline"]
    results = []
    for r in (2, 3):
        for combo in combinations(methods, r):
            union = set().union(*(hit_sets[m] for m in combo))
            results.append((combo, len(union)))
    return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]


# ── Diversity ─────────────────────────────────────────────────────────────────

def _hit_diversity(smiles_list: list[str]) -> float:
    """
    Chemical diversity among a set of hits: 1 - mean pairwise ECFP4 Tanimoto.
    Returns 0.0 for sets of size < 2.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]
    if len(mols) < 2:
        return 0.0
    fps  = [_ecfp4(m) for m in mols]
    sims = [
        DataStructs.TanimotoSimilarity(fps[i], fps[j])
        for i in range(len(fps))
        for j in range(i + 1, len(fps))
    ]
    return 1.0 - float(np.mean(sims))


# ── Summary with multi-threshold + stratification ─────────────────────────────

def summarize(
    df:          pd.DataFrame,
    tau_t_list:  list[float] = DEFAULT_TAU_T_LIST,
    tau_d:       float       = DEFAULT_TAU_D,
) -> pd.DataFrame:
    """
    Per-method summary stratified by quality bin, reported at multiple tau_t values.

    For each (method, quality_bin, tau_t) combination:
      - n_variants, n_hits, hit_rate
      - n_unique_hits (deduplicated by canonical SMILES)
      - expansion_factor vs "baseline" method at the same bin and threshold
      - mean_tanimoto, mean_desirability
      - diversity (1 - mean pairwise ECFP4 Tanimoto among hits)

    Args:
        df:         Output of score_batch() or pd.concat of multiple batches.
                    Must contain columns: method, quality_bin, tanimoto, desirability, variant_smiles.
        tau_t_list: List of substructural conservation thresholds to report at.
        tau_d:      Physicochemical conservation threshold.

    Returns:
        DataFrame sorted by (tau_t, quality_bin, method).
    """
    rows = []

    for tau_t in tau_t_list:
        df_t = df.copy()
        df_t["is_hit"] = classify_hits(df_t, tau_t, tau_d)

        # Pre-compute baseline unique hits per bin for expansion factor
        baseline_hits_per_bin: dict[str, set[str]] = {}
        if "baseline" in df_t["method"].values:
            for qbin in BIN_LABELS:
                grp = df_t[(df_t["method"] == "baseline") & (df_t["quality_bin"] == qbin)]
                baseline_hits_per_bin[qbin] = set(grp.loc[grp["is_hit"], "variant_smiles"])

        for (method, qbin), grp in df_t.groupby(["method", "quality_bin"]):
            hits         = grp[grp["is_hit"]]
            unique_hits  = set(hits["variant_smiles"])
            baseline_set = baseline_hits_per_bin.get(qbin, set())
            ef           = expansion_factor(unique_hits, baseline_set) if method != "baseline" else 1.0
            diversity    = _hit_diversity(list(unique_hits))

            rows.append({
                "method":            method,
                "quality_bin":       qbin,
                "tau_t":             tau_t,
                "tau_d":             tau_d,
                "n_variants":        len(grp),
                "n_hits":            len(hits),
                "hit_rate":          round(len(hits) / len(grp), 4) if len(grp) > 0 else 0.0,
                "n_unique_hits":     len(unique_hits),
                "expansion_factor":  round(ef, 4),
                "mean_tanimoto":     round(grp["tanimoto"].mean(), 4),
                "mean_desirability": round(grp["desirability"].mean(), 4),
                "diversity":         round(diversity, 4),
            })

    return (
        pd.DataFrame(rows)
        .sort_values(["tau_t", "quality_bin", "method"])
        .reset_index(drop=True)
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Score variant molecules against property specifications (scoring_v2)"
    )
    parser.add_argument(
        "--input", type=pathlib.Path, required=True,
        help=(
            "JSON file: list of {spec_smiles, baseline_quality, "
            "methods: {method_name: [smiles, ...], ...}}"
        ),
    )
    parser.add_argument("--output",    type=pathlib.Path, default=pathlib.Path("data/scores_v2.csv"))
    parser.add_argument("--summary",   type=pathlib.Path, default=pathlib.Path("data/summary_v2.csv"))
    parser.add_argument("--tau-t",     type=float, nargs="+", default=DEFAULT_TAU_T_LIST)
    parser.add_argument("--tau-d",     type=float, default=DEFAULT_TAU_D)
    parser.add_argument("--espsim",    action="store_true",
                        help="Compute ESPsim (requires espsim package; generates 3D conformers)")
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))

    all_dfs: list[pd.DataFrame] = []
    t0 = time.time()

    for entry in data:
        spec = make_spec(entry["spec_smiles"], generate_conformer=args.espsim)
        if spec is None:
            print(f"[WARN] Invalid spec SMILES: {entry['spec_smiles'][:60]}")
            continue

        bq = float(entry["baseline_quality"])

        for method_name, variant_list in entry["methods"].items():
            batch_df = score_batch(variant_list, spec, bq, method_name, args.espsim)
            if not batch_df.empty:
                all_dfs.append(batch_df)

    elapsed = time.time() - t0

    if not all_dfs:
        print("No valid variants scored.")
    else:
        df = pd.concat(all_dfs, ignore_index=True)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Scored {len(df)} variants in {elapsed:.1f}s → {args.output}")

        summary = summarize(df, args.tau_t, args.tau_d)
        summary.to_csv(args.summary, index=False)
        print(f"Summary ({len(summary)} rows) → {args.summary}")

        print("\n── Summary (first 20 rows) ────────────────────────────────")
        print(summary.head(20).to_string(index=False))

        # Complementarity at first tau_t
        tau_t0 = args.tau_t[0]
        df["is_hit"] = classify_hits(df, tau_t0, args.tau_d)
        hit_sets = {
            m: set(g.loc[g["is_hit"], "variant_smiles"])
            for m, g in df.groupby("method")
        }
        comp_df = complementarity(hit_sets)
        print(f"\n── Complementarity (Jaccard, tau_t={tau_t0}) ─────────────")
        print(comp_df.round(3).to_string())

        combos = best_combinations(hit_sets)
        print(f"\n── Best 2/3-method combinations by unique-hit coverage ────")
        for combo, n in combos:
            print(f"  {' + '.join(combo):40s}  {n} unique hits")
