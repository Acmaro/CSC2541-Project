"""Smoke test for scoring_v2.py — run with: conda run -n prexsyn python src/evaluation/_test_scoring_v2.py"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from src.evaluation.scoring_v2 import (
    make_spec, score_batch, summarize, complementarity, best_combinations,
    expansion_factor, classify_hits,
)

# ── Build spec from aspirin ───────────────────────────────────────────────────
spec = make_spec("CC(=O)Oc1ccccc1C(=O)O")
assert spec is not None, "make_spec failed"
print(f"Spec  MW={spec.mw:.1f}  CLogP={spec.clogp:.2f}  TPSA={spec.tpsa:.1f}  "
      f"HBD={spec.hbd}  HBA={spec.hba}  RotBonds={spec.rot_bonds}")

# ── Score variants for three methods ─────────────────────────────────────────
variants_crem     = ["CC(=O)Oc1ccccc1C(=O)OC", "CC(=O)Oc1ccc(F)cc1C(=O)O", "CC(=O)Nc1ccccc1C(=O)O"]
variants_mmpdb    = ["CC(=O)Oc1ccccc1C(=O)N",  "ClC(=O)Oc1ccccc1C(=O)O"]
variants_baseline = ["CC(=O)Oc1ccccc1C(=O)O",  "CC(=O)Oc1ccc(Cl)cc1C(=O)O"]

df = pd.concat([
    score_batch(variants_crem,     spec, baseline_quality=0.75, method="CReM"),
    score_batch(variants_mmpdb,    spec, baseline_quality=0.75, method="mmpdb"),
    score_batch(variants_baseline, spec, baseline_quality=0.75, method="baseline"),
], ignore_index=True)

print("\n── Per-variant scores ───────────────────────────────────────────────────")
print(df[["method", "variant_smiles", "tanimoto", "desirability", "quality_bin"]].to_string(index=False))

# ── Desirability components ───────────────────────────────────────────────────
print("\n── Desirability components ──────────────────────────────────────────────")
print(df[["method", "mw_score", "clogp_score", "tpsa_score", "hbd_score", "hba_score", "rot_bonds_score", "desirability"]].to_string(index=False))

# ── Summary (multi-threshold + bin stratification) ────────────────────────────
print("\n── Summary (multi-threshold) ────────────────────────────────────────────")
summary = summarize(df)
print(summary[["method", "quality_bin", "tau_t", "n_variants", "n_hits", "hit_rate", "expansion_factor", "diversity"]].to_string(index=False))

# ── Complementarity ───────────────────────────────────────────────────────────
df2 = df.copy()
df2["is_hit"] = classify_hits(df2, tau_t=0.6, tau_d=0.8)
hit_sets = {m: set(g.loc[g["is_hit"], "variant_smiles"]) for m, g in df2.groupby("method")}
print("\n── Hit sets ─────────────────────────────────────────────────────────────")
for m, s in hit_sets.items():
    print(f"  {m}: {len(s)} hits — {s}")

print("\n── Complementarity matrix (Jaccard) ─────────────────────────────────────")
print(complementarity(hit_sets).round(3))

print("\n── Best 2/3-method combinations ────────────────────────────────────────")
for combo, n in best_combinations(hit_sets):
    print(f"  {' + '.join(combo):30s}  {n} unique hits")

# ── Assertions ────────────────────────────────────────────────────────────────
assert "tanimoto" in df.columns
assert "desirability" in df.columns
assert "quality_bin" in df.columns
assert all(df["quality_bin"] == "0.7-0.85"), f"Expected bin 0.7-0.85, got {df['quality_bin'].unique()}"
assert "expansion_factor" in summary.columns
assert not summary.empty

print("\nAll assertions passed.")
