---
title: "Working Notes"
subtitle: "PrexSyn+ — Internal Reference"
date: March 2026
geometry: margin=1in
fontsize: 11pt
---

# Project Summary

Benchmark five molecular modification methods (CReM, mmpdb, JT-VAE, LibINVENT, ReactEA) layered on PrexSyn output, across 1,000 ChEMBL-derived property specs. Core outputs: individual method rankings by marginal value over a repeated-sampling baseline, and a complementarity matrix identifying optimal multi-method combinations. Course project for CSC2541, University of Toronto.

# Design Decisions

**ChEMBL as spec generator, not target.** We extract property profiles from ChEMBL molecules and use them as input specs. The molecules themselves are not optimization targets. Structural similarity to the ChEMBL reference is reported only as a diagnostic (proximity to known bioactive space).

**Repeated-sampling baseline.** The baseline is $N$ additional PrexSyn draws from the same spec. Its structural limitation — confinement to Enamine REAL — is exactly what the benchmark tests. Key sub-question: does repeated sampling saturate while modification opens new territory?

**Multi-level property conservation.** Three complementary metrics: ECFP4 Tanimoto (substructural conservation), physicochemical desirability (geometric mean of per-descriptor scores), and ESPsim (3D shape + electrostatic similarity).

**Parameterized thresholds.** Hit-classification thresholds ($\tau_d$, $\tau_t$) are parameters. We report at multiple values (Tanimoto $\geq$ 0.6, 0.7, 0.85) to avoid cutoff artifacts.

**Single best candidate as seed ($K=1$).** Cleaner attribution to the modification method. Top-$K$ seeding is a natural extension.

**Modify all specs, stratify by difficulty.** All 1,000 specs proceed to modification regardless of PrexSyn baseline quality. Results are stratified by seed Tanimoto (<0.5, 0.5–0.7, 0.7–0.85, 0.85–1.0). This avoids bias toward hard cases while revealing when modification helps most.

**Complementarity analysis.** Pairwise hit-set overlap identifies redundant vs. complementary methods. We report optimal 2- and 3-method combinations for maximum unique coverage.

**Terminology.** "Post-generation molecular modification" — broad, precise, avoids the narrowness of "mutation."

# Open Questions

1. **Property extraction.** ECFP4 fingerprints (PrexSyn's native input) vs. full descriptor set from PrexSyn Table 3?
2. **Conservation thresholds.** Tanimoto: 0.7 (virtual screening standard) or 0.85 (stricter)? Desirability geometric mean: 0.8? Report at multiple values; need principled defaults.
3. **Synthesizability depth.** AiZynthFinder depth 1 (faster, 100/1000 ChEMBL) or depth 2 (catches more, 171/1000, ~5x slower)? Possible: depth 1 primary, depth 2 sensitivity check.
4. **Method tractability.** Are all 5 methods tractable at 1,000 specs $\times$ $N$ variants each?
5. **Variants per molecule ($N$).** $N = 1000$ discussed. Total evaluation load: 1,000 $\times$ 5 $\times$ 1,000 = 5M variants through AiZynthFinder. Feasible?
6. **ESPsim conformer dependency.** Using ChEMBL molecule's ETKDG conformer as 3D reference is pragmatic but couples evaluation to the ChEMBL molecule beyond its property values. Accept this, or demote ESPsim to secondary-only?
7. **Diversity computation.** Mean pairwise Tanimoto is $O(n^2)$. May need sampling-based approximation for large hit sets.
8. **Stratification bins.** Current bins (<0.5, 0.5–0.7, 0.7–0.85, 0.85–1.0) are provisional. Adjust after observing actual baseline quality distribution.
9. **Complementarity reporting.** 10 pairs, 10 triples for 5 methods — report all or top-$K$?

# Future Directions

**Composite pipelines.** Chain the most effective, most complementary operators: conservative local edits (mmpdb) -- medium-scale fragment substitution (CReM) -- global latent-space exploration (JT-VAE), with scoring and synthesis filtering at every stage. An orchestration layer (e.g., Deriver or a custom agent framework) would manage the cascade. No public system integrates these operators under unified synthesis constraints. Our complementarity analysis provides the empirical prerequisite.

**Top-$K$ seeding.** Seed modification from PrexSyn's top $K$ candidates instead of $K=1$. Tests whether seed diversity compounds with method diversity.

**CReM context radius as independent variable.** Benchmark CReM at radius = 1, 2, 3 to produce a clean edit-distance-vs-hit-rate curve within a single tool.

**Expected outcomes.** (a) Modification methods likely outperform repeated sampling on diversity, since they can escape Enamine REAL. (b) Rule-based methods (CReM, mmpdb) probably maintain higher synthesizability rates than learned methods (JT-VAE). (c) Methods at different edit scales are probably complementary (low overlap). (d) Repeated PrexSyn sampling may saturate — diminishing returns on diversity — while modification does not.

# Discarded Ideas

**RECAP** — retrosynthetic fragment recombination, built into RDKit. Subsumed by CReM, which does the same operations with context constraints and a real-molecule fragment database.

**Heterocycle Isostere Explorer** — ring-system replacement. Subsumed by CReM (ring replacement is a special case of fragment mutation at ring-containing sites).

**Notable alternatives considered but not selected:**

| Method | Why considered | Why not selected |
|--------|---------------|-----------------|
| Modof | Fragment-level latent editing; chains edits | JT-VAE is more established; swap if JT-VAE proves problematic |
| Gargoyles | Fragment + atom RL with built-in AiZynthFinder | More complex setup; less community adoption |
| He et al. MMP Transformer | Learned MMP with multi-property control | mmpdb is more mature and turnkey |
| Deriver | Orchestration framework wrapping generators | A framework, not a method — relevant to pipeline engineering |
| MIMOSA | Graph-based MCMC with ring-level proposals | Narrower than CReM; less adoption |

# Stretch Metrics

- **Gobbi_Pharm2D** (RDKit 3D pharmacophore fingerprints): Pharmacophore-aware comparison. Low effort.
- **CLAMP embeddings** (cosine similarity): Only pretrained embedding that consistently beats ECFP4 in 2025 benchmarks. Moderate effort.
- **SALI** (Structure-Activity Landscape Index): Identifies activity cliffs where all similarity metrics fail. Adds analytical depth. Requires known bioactivity data.

# Research Provenance

**Undermind search (2026-03-05).** Systematic literature survey of ~50 molecular modification methods across 5 categories (scaffold-constrained decoration, latent-space editing, MMP/bioisosteric transforms, fragment-based recombination, ring/heterocycle replacement). Full results in `researching-mod-algos/`. Key finding: CReM dominates recommendations across multiple categories, subsuming RECAP and Heterocycle Isostere Explorer. No public tool integrates these methods under unified synthesis constraints — validates our benchmarking premise.

**Foundational references by corpus citation rate:** Gomez-Bombarelli VAE (0.40), JT-VAE (0.37), REINVENT (0.32), SMILES encoding (0.31), QED (0.23).
