---
title: "Expanding Synthesizable Chemical Space: Benchmarking Post-Generation Molecular Modification as a Complement to Generative AI"
author:
  - Tyler Biswurm
  - Jingxin Huang
  - Yuma Nakamura
  - Ben Luo
date: March 2026
geometry: margin=1in
fontsize: 11pt
linestretch: 1.15
header-includes:
  - \usepackage{booktabs}
  - \usepackage{amsmath}
---

# Abstract

Synthesis-aware generative models produce drug-like molecules guaranteed to be synthesizable from commercial building blocks, but each is bounded by its training chemical space --- a constraint inherent to the paradigm. Retraining on a larger space shifts this boundary; repeated sampling cannot reach beyond it. Here we ask whether classical molecular modification methods, which operate without such constraints, expand the space of viable candidates when composed with a frontier generative model.

We layer five modification methods on top of PrexSyn, the current state-of-the-art synthesis-aware generator. For 1,000 property specifications derived from ChEMBL bioactive compounds, PrexSyn produces an initial candidate; each method then generates structural variants. Variants are filtered for synthesizability via retrosynthetic analysis and scored for property conservation using substructural, physicochemical, and 3D shape/electrostatic metrics. We compare each method against a repeated-sampling baseline, stratify results by specification difficulty, and compute pairwise method complementarity to identify optimal combinations.

\bigskip

# 1. Introduction

Drug discovery frequently begins with a property wish list: a target molecular weight, lipophilicity, polar surface area, and related descriptors that define a pharmacological profile of interest. The practical task is to find synthesizable molecules that match it. Synthesis-aware generative models address this directly --- they accept a property specification and output molecules composed of purchasable building blocks via validated reaction templates, guaranteeing synthesizability by construction. Several models operate in this paradigm, including SynNet, ChemProjector, SynFormer, and PrexSyn.

PrexSyn, the current state of the art, generates property-conditioned molecules within the Enamine REAL chemical space at high throughput (~0.26 s per target) using ~480,000 in-stock building blocks and 115 reaction templates. But Enamine REAL covers a strict subset of drug-like chemistry: PrexSyn reconstructs 94% of molecules drawn from this space but only 28% from the broader ChEMBL distribution (a curated database of bioactive compounds with known pharmacological activity), indicating substantial pharmacologically relevant territory beyond its reach. This coverage gap is structural --- any synthesis-aware model is bounded by its building block library and template set. Retraining on a larger space moves the boundary but does not eliminate it, and demands substantial data curation.

We investigate a complementary strategy: applying post-generation molecular modification to the generative model's output. Methods such as fragment substitution, matched molecular pair transforms, and latent-space perturbation predate generative AI and are not constrained to Enamine REAL. We benchmark five such methods --- three rule-based, two learned --- on their ability to produce synthesizable, property-conserving candidates beyond what repeated PrexSyn sampling achieves. Results are stratified by specification difficulty to reveal when modification adds the most value, and a complementarity analysis identifies which method combinations maximize total coverage. Our benchmark yields two core outputs: individual method rankings by marginal value (Table 1) and optimal multi-method combinations (Table 2) --- results applicable to both manual workflows and automated agent-driven molecular design systems.

# 2. Methods

## 2.1 Modification methods

We evaluate five open-source methods that span the rule-based/learned axis and the local-to-global edit-scale axis (Fig. 1).

**CReM** (Polishchuk, 2020) performs fragment-level mutate, grow, and link operations using a context-constrained fragment database mined from ChEMBL. A tunable context radius controls edit locality: small radii produce conservative R-group tweaks; larger radii enable ring-system swaps and scaffold changes.

**mmpdb** mines matched molecular pair transformation rules from compound datasets and applies single-site substitutions to new molecules (e.g., methyl $\to$ chloro, phenyl $\to$ pyridyl). Rules carry empirical property deltas, enabling selection of transformations likely to preserve target properties. It is the standard tool for conservative, chemist-interpretable local edits.

**JT-VAE** (Jin et al., 2018) encodes molecules as junction-tree decompositions into a continuous latent space. Modifications are performed by perturbing or interpolating latent vectors, then decoding. It is the most widely cited latent-space molecular editor.

**LibINVENT** (via REINVENT4) generates novel R-group decorations on a fixed scaffold using reinforcement learning steered toward desired properties. In our pipeline, the scaffold derives from PrexSyn's output.

**ReactEA** is an evolutionary algorithm over 22,949 biochemical reaction rules. Every output carries an explicit synthetic route, making it the most synthesis-aligned method in the set.

The core three (CReM, mmpdb, JT-VAE) were selected for coverage across edit scales --- local (mmpdb), tunable local-to-medium (CReM), and medium-to-global (JT-VAE) --- and across paradigms. LibINVENT and ReactEA extend the benchmark to scaffold-constrained decoration and synthesis-enforced search. Selection was informed by a systematic review of ~50 candidate methods.

## 2.2 Benchmarking pipeline

Property specifications are derived from 1,000 ChEMBL bioactive compounds. For each compound we extract an ECFP4 fingerprint, physicochemical descriptors (MW, CLogP, TPSA, HBD, HBA, rotatable bonds), and a 3D conformer via ETKDG. The molecular identity is retained only for diagnostic purposes.

For each specification, PrexSyn generates 256 candidates. The top candidate by ECFP4 Tanimoto to the spec serves as the seed for modification. Each of the five methods produces $N$ structural variants from this seed. As a baseline, PrexSyn draws $N$ additional samples from the same specification within Enamine REAL. All variants pass through the evaluation pipeline described below.

Results are stratified by PrexSyn's baseline quality (the seed's Tanimoto to the spec, binned as $<$0.5, 0.5--0.7, 0.7--0.85, 0.85--1.0) to reveal the conditions under which modification adds the most marginal value. A detailed pseudocode specification appears in Appendix A.

## 2.3 Evaluation

Variants pass through two sequential gates. First, **synthesizability**: AiZynthFinder checks whether each variant can be retrosynthetically decomposed into purchasable Enamine building blocks via known reactions. Variants without a valid route are discarded. Second, **property conservation**, assessed on two complementary axes:

- *Substructural conservation* (ECFP4 Tanimoto to the specification fingerprint). Measures shared chemical features. Recent benchmarks show this metric matches or exceeds pretrained molecular embeddings for property prediction.

- *Physicochemical profile conservation*. Per-descriptor deltas mapped through tolerance functions to 0--1 desirability scores, aggregated by geometric mean. A single badly violated descriptor drives the aggregate down. These are the same descriptors PrexSyn conditions on, ensuring evaluation matches the model's abstraction level.

A variant is a **hit** if both scores exceed parameterized thresholds ($\tau_t$, $\tau_d$). We report results at multiple threshold values to avoid conclusions tied to a single cutoff.

As a supplementary metric, we compute **ESPsim** (shape + electrostatic potential similarity) using a 3D conformer of the ChEMBL reference as a proxy for the desired binding-site shape. This captures complementarity that 2D metrics miss but introduces a dependency on the reference molecule's geometry. We also report structural similarity to the ChEMBL reference as a diagnostic distribution (proximity to known bioactive space), not a success criterion.

Per method, we report hit rate, unique hits, expansion factor (unique hits relative to baseline), chemical diversity ($1 - $ mean pairwise Tanimoto among hits), and wall-clock compute cost. Pairwise method complementarity --- the fraction of each method's hits not found by any other --- identifies the two- and three-method combinations that maximize total unique coverage.

# 3. Scope and limitations

Property conservation is assessed computationally at the same abstraction level as PrexSyn's inputs (computed descriptors). This is standard practice but does not substitute for experimental validation. Synthesizability is checked via retrosynthetic analysis against Enamine building blocks; molecules synthesizable through alternative routes may be missed. The method shortlist is representative, not exhaustive. Our complementarity analysis provides the empirical prerequisite for composite modification pipelines that chain operators at different edit scales under unified synthesis constraints --- a natural next step.

\newpage

# Appendix A: Detailed Pipeline Pseudocode

```
STAGE 1: Derive property specifications from ChEMBL
------------------------------------------------------------

Input:  D_chembl = {m_1, ..., m_1000}

for each molecule m_i in D_chembl:
    spec_i = ExtractPropertySpec(m_i)
        # fingerprint:   ECFP4 bitvector
        # descriptors:   MW, CLogP, TPSA, HBD, HBA, RotBonds, AromaticRings
        # conformer:     3D coordinates via ETKDG
    store spec_i
    store m_i as diagnostic_ref_i      # NOT an optimization target

STAGE 2: Generate initial candidates via PrexSyn
------------------------------------------------------------

for each spec_i:
    candidates_i = PrexSyn.generate(spec_i, num_samples=256)
    seed_i = argmax_{c in candidates_i} ECFP4_Tanimoto(c, spec_i.fp)
    baseline_quality_i = ECFP4_Tanimoto(seed_i, spec_i.fp)
    pass (spec_i, seed_i, baseline_quality_i) to Stage 3

STAGE 3: Apply post-generation molecular modification
------------------------------------------------------------

Methods = {CReM, mmpdb, JT-VAE, LibINVENT, ReactEA}

for each (spec_i, seed_i, baseline_quality_i):
    for each method M in Methods:
        variants_M_i = M.modify(seed_i, num_variants=N)
    variants_baseline_i = PrexSyn.generate(spec_i, num_samples=N)

STAGE 4: Multi-criteria evaluation
------------------------------------------------------------

for each variant v:

    # Gate 1: Synthesizability
    if not AiZynthFinder.route_found(v, building_blocks="enamine"):
        discard v; continue

    # Gate 2: Property conservation
    substruct_v = ECFP4_Tanimoto(v, spec_i.fp)

    for each descriptor d in {MW, CLogP, TPSA, HBD, HBA, RotBonds}:
        desirability_d = max(0, 1 - |Descriptor_d(v) - spec_i.d| / tolerance_d)
    desirability_v = GeometricMean(desirability_d for all d)

    espsim_v = ESPSim.similarity(v, diagnostic_ref_i.conformer)

    # Hit classification
    if desirability_v >= tau_d AND substruct_v >= tau_t:
        mark v as HIT

    # Diagnostic
    bioactive_proximity_v = ECFP4_Tanimoto(v, diagnostic_ref_i)

STAGE 5: Aggregation, stratification, and complementarity
------------------------------------------------------------

for each method M (including baseline):
    HitRate_M        = |hits_M| / |synthesizable_variants_M|
    UniqueHits_M     = |deduplicated hits_M|
    Expansion_M      = UniqueHits_M / UniqueHits_baseline
    Diversity_M      = 1 - MeanPairwiseTanimoto(hits_M)
    ComputeCost_M    = wall-clock seconds per input molecule
rank by Expansion_M (primary), HitRate_M, Diversity_M

# Stratification
bins = {<0.5, 0.5-0.7, 0.7-0.85, 0.85-1.0}
for each bin b:
    report all metrics restricted to specs with baseline_quality in b

# Complementarity
for each pair (M_i, M_j):
    overlap_ij = |hits_i AND hits_j| / |hits_i OR hits_j|
for each triple (M_i, M_j, M_k):
    combined = |hits_i OR hits_j OR hits_k|
report pairwise overlap matrix; top combinations by total unique hits
```
