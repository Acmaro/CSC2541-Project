"""Generate notebooks/pipeline_analysis.ipynb from scratch."""
import json
import pathlib
import uuid

ROOT = pathlib.Path(r"D:\AI4DD Project\CSC2541-Project")
OUT  = ROOT / "notebooks" / "pipeline_analysis.ipynb"


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": src,
        "outputs": [],
        "execution_count": None,
    }


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": src,
    }


cells = []

# ─── Title ────────────────────────────────────────────────────────────────────
cells.append(md(
    "# Pipeline Analysis\n\n"
    "**Stage 5 of the benchmarking pipeline**: aggregates all method outputs\n"
    "and produces Table 1 (per-method metrics) and Table 2 (complementarity).\n\n"
    "Follows Appendix A pseudocode exactly.  "
    "Run *after* all method notebooks have finished.\n\n"
    "```\n"
    "Stage 1  Derive specs      <- prexsyn_baseline.ipynb  -> seeds_for_methods.json\n"
    "Stage 2  PrexSyn seeds     <- prexsyn_baseline.ipynb\n"
    "Stage 3  Method variants   <- per-method notebooks    -> *_scores.csv\n"
    "Stage 4  Evaluation gates  <- per-method notebooks    -> *_synth_checkpoint.json\n"
    "Stage 5  Aggregation       <- THIS NOTEBOOK           -> pipeline_results/\n"
    "```"
))

# ─── Configuration ────────────────────────────────────────────────────────────
cells.append(md("## Configuration"))

cells.append(code(
    "import json\n"
    "import pathlib\n"
    "import sys\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "from tqdm.notebook import tqdm\n"
    "\n"
    "ROOT     = pathlib.Path('..')\n"
    "DATA_DIR = ROOT / 'data'\n"
    "OUT_DIR  = DATA_DIR / 'pipeline_results'   # <- all outputs written here\n"
    "OUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "sys.path.insert(0, str(ROOT))\n"
    "from src.evaluation.scoring_v2 import (\n"
    "    classify_hits, summarize, complementarity, best_combinations,\n"
    "    DEFAULT_TAU_T_LIST, DEFAULT_TAU_D,\n"
    ")\n"
    "\n"
    "TAU_T_LIST = DEFAULT_TAU_T_LIST   # [0.6, 0.7, 0.85]  <- paper §2.3\n"
    "TAU_D      = DEFAULT_TAU_D        # 0.8\n"
    "\n"
    "# Method registry --------------------------------------------------\n"
    "# Add new methods here as their notebooks are completed.\n"
    "# Each entry needs:\n"
    "#   scores:     CSV from scoring_v2.score_batch() with column 'variant_smiles'\n"
    "#   synth_ckpt: JSON {smiles: is_solved} from AiZynthFinder gate\n"
    "METHODS = {\n"
    "    'baseline': {\n"
    "        'scores':     DATA_DIR / 'baseline_scores.csv',\n"
    "        'synth_ckpt': DATA_DIR / 'baseline_synth_checkpoint.json',\n"
    "    },\n"
    "    'LibINVENT': {\n"
    "        'scores':     DATA_DIR / 'libinvent_runs/prexsyn_decoration/libinvent_scores.csv',\n"
    "        'synth_ckpt': DATA_DIR / 'libinvent_runs/prexsyn_decoration/synth_checkpoint.json',\n"
    "    },\n"
    "    # 'CReM':    {'scores': DATA_DIR / 'crem_scores.csv',    'synth_ckpt': DATA_DIR / 'crem_synth_checkpoint.json'},\n"
    "    # 'mmpdb':   {'scores': DATA_DIR / 'mmpdb_scores.csv',   'synth_ckpt': DATA_DIR / 'mmpdb_synth_checkpoint.json'},\n"
    "    # 'JT-VAE':  {'scores': DATA_DIR / 'jtvae_scores.csv',   'synth_ckpt': DATA_DIR / 'jtvae_synth_checkpoint.json'},\n"
    "    # 'ReactEA': {'scores': DATA_DIR / 'reactea_scores.csv', 'synth_ckpt': DATA_DIR / 'reactea_synth_checkpoint.json'},\n"
    "}\n"
    "\n"
    "SUMMARY_CSV    = OUT_DIR / 'summary.csv'            # Table 1\n"
    "COMPLEMENT_CSV = OUT_DIR / 'complementarity.csv'    # Table 2 pairwise\n"
    "COMBOS_CSV     = OUT_DIR / 'best_combinations.csv'  # Table 2 top combos\n"
    "\n"
    "print(f'Output dir    : {OUT_DIR}')\n"
    "print(f'Methods       : {list(METHODS)}')\n"
    "print(f'Thresholds    : tau_t={TAU_T_LIST}, tau_d={TAU_D}')\n"
))

# ─── Stage 1 ──────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Stage 1 — Property Specifications (from ChEMBL)\n\n"
    "**Pseudocode**: `spec_i = ExtractPropertySpec(m_i)` — ECFP4, MW, CLogP, TPSA,\n"
    "HBD, HBA, RotBonds, 3D conformer via ETKDG.\n\n"
    "Specifications were derived by `prexsyn_baseline.ipynb` and written to\n"
    "`seeds_for_methods.json`.  We load them here for quality-bin stratification."
))

cells.append(code(
    "assert (DATA_DIR / 'seeds_for_methods.json').exists(), (\n"
    "    'seeds_for_methods.json missing. Run prexsyn_baseline.ipynb first.'\n"
    ")\n"
    "\n"
    "with open(DATA_DIR / 'seeds_for_methods.json') as f:\n"
    "    seeds = json.load(f)\n"
    "\n"
    "seeds_df = pd.DataFrame([\n"
    "    {\n"
    "        'spec_smiles':      e['spec_smiles'],\n"
    "        'seed_smiles':      e['seed_smiles'],\n"
    "        'baseline_quality': e['baseline_quality'],\n"
    "        'quality_bin':      e['quality_bin'],\n"
    "    }\n"
    "    for e in seeds\n"
    "])\n"
    "\n"
    "print(f'Specs loaded: {len(seeds_df)}')\n"
    "print('\\nQuality-bin counts:')\n"
    "print(seeds_df['quality_bin'].value_counts().sort_index().to_string())\n"
))

# ─── Stage 2 ──────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Stage 2 — PrexSyn Seeds & Baseline Quality\n\n"
    "**Pseudocode**: `seed_i = argmax Tanimoto(c, spec_i.fp)` over 256 candidates.\n"
    "`baseline_quality_i = Tanimoto(seed_i, spec_i.fp)`.\n\n"
    "The seed Tanimoto defines difficulty: specs with low `baseline_quality` are\n"
    "harder for PrexSyn — modification may add more value in those bins."
))

cells.append(code(
    "fig, ax = plt.subplots(figsize=(6, 3))\n"
    "ax.hist(seeds_df['baseline_quality'], bins=20, edgecolor='white')\n"
    "for x in (0.5, 0.7, 0.85):\n"
    "    ax.axvline(x, color='gray', linestyle='--', linewidth=0.8)\n"
    "ax.set_xlabel('Baseline quality (Tanimoto seed to spec)')\n"
    "ax.set_ylabel('Specs')\n"
    "ax.set_title('PrexSyn seed quality distribution')\n"
    "plt.tight_layout()\n"
    "plt.savefig(OUT_DIR / 'baseline_quality_dist.png', dpi=150)\n"
    "plt.show()\n"
    "print(seeds_df['baseline_quality'].describe().round(3).to_string())\n"
))

# ─── Stage 3 ──────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Stage 3 — Load Method Variants\n\n"
    "**Pseudocode**: `variants_M_i = M.modify(seed_i, num_variants=N)`.\n\n"
    "Each method notebook writes a scored CSV.  We load all registered methods;\n"
    "missing files are skipped with a warning."
))

cells.append(code(
    "per_method: dict[str, pd.DataFrame] = {}\n"
    "\n"
    "for name, cfg in METHODS.items():\n"
    "    path = cfg['scores']\n"
    "    if not path.exists():\n"
    "        print(f'[SKIP] {name}: {path.name} not found')\n"
    "        continue\n"
    "    df = pd.read_csv(path)\n"
    "    df['method'] = name       # normalise label\n"
    "    per_method[name] = df\n"
    "    print(f'[OK]   {name}: {len(df):,} rows')\n"
    "\n"
    "assert per_method, 'No method data found. Run method notebooks first.'\n"
))

# ─── Stage 4 ──────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Stage 4 — Evaluation Gates\n\n"
    "**Pseudocode**:\n"
    "```\n"
    "# Gate 1: Synthesizability\n"
    "if not AiZynthFinder.route_found(v): discard v\n"
    "\n"
    "# Gate 2: Property conservation\n"
    "substruct_v    = ECFP4_Tanimoto(v, spec_i.fp)               # tau_t\n"
    "desirability_v = GeometricMean(per-descriptor tolerances)   # tau_d\n"
    "\n"
    "if desirability_v >= tau_d AND substruct_v >= tau_t: mark v as HIT\n"
    "```\n\n"
    "`is_synth` is joined from each method's AiZynthFinder checkpoint.\n"
    "Hit classification is re-applied uniformly at all `TAU_T_LIST` values."
))

cells.append(code(
    "# Gate 1: attach is_synth from each checkpoint\n"
    "for name, df in per_method.items():\n"
    "    ckpt_path = METHODS[name]['synth_ckpt']\n"
    "    if ckpt_path.exists():\n"
    "        with open(ckpt_path) as f:\n"
    "            ckpt: dict[str, bool] = json.load(f)\n"
    "        df['is_synth'] = df['variant_smiles'].map(ckpt).fillna(False)\n"
    "        print(f'{name}: {df[\"is_synth\"].sum()}/{len(df)} synthesizable')\n"
    "    else:\n"
    "        # Baseline: only synth-passing variants were scored -> all True\n"
    "        df['is_synth'] = True\n"
    "        print(f'{name}: no checkpoint -> is_synth=True (baseline assumption)')\n"
    "\n"
    "# Merge; restrict to synthesizable variants for all metrics\n"
    "all_df   = pd.concat(per_method.values(), ignore_index=True)\n"
    "synth_df = all_df[all_df['is_synth']].reset_index(drop=True)\n"
    "\n"
    "print(f'\\nAll variants  : {len(all_df):,}')\n"
    "print(f'Synthesizable : {len(synth_df):,} ({100*len(synth_df)/max(len(all_df),1):.1f}%)')\n"
    "print(synth_df['method'].value_counts().to_string())\n"
))

# ─── Stage 5a ─────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Stage 5a — Per-Method Metrics (Table 1)\n\n"
    "**Pseudocode**:\n"
    "```\n"
    "HitRate_M    = |hits_M| / |synthesizable_variants_M|\n"
    "UniqueHits_M = |deduplicated hits_M|\n"
    "Expansion_M  = UniqueHits_M / UniqueHits_baseline\n"
    "Diversity_M  = 1 - MeanPairwiseTanimoto(hits_M)\n"
    "```\n\n"
    "`summarize()` computes all metrics at every `tau_t`, stratified by `quality_bin`."
))

cells.append(code(
    "summary_df = summarize(synth_df, tau_t_list=TAU_T_LIST, tau_d=TAU_D)\n"
    "summary_df.to_csv(SUMMARY_CSV, index=False)\n"
    "print(f'Table 1 -> {SUMMARY_CSV}  ({len(summary_df)} rows)')\n"
    "display(summary_df.sort_values(['tau_t', 'quality_bin', 'method']))\n"
))

# ─── Stage 5b ─────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Stage 5b — Stratification by Quality Bin\n\n"
    "**Pseudocode**: `for each bin b: report metrics restricted to specs in b`\n\n"
    "Hit rates by bin reveal when modification adds the most marginal value."
))

cells.append(code(
    "BINS_ORDERED = ['<0.5', '0.5-0.7', '0.7-0.85', '0.85-1.0']\n"
    "n_thresh = len(TAU_T_LIST)\n"
    "\n"
    "fig, axes = plt.subplots(1, n_thresh, figsize=(5 * n_thresh, 4), sharey=True)\n"
    "if n_thresh == 1:\n"
    "    axes = [axes]\n"
    "\n"
    "for ax, tau_t in zip(axes, TAU_T_LIST):\n"
    "    sub = summary_df[summary_df['tau_t'] == tau_t].copy()\n"
    "    sub['quality_bin'] = pd.Categorical(sub['quality_bin'], categories=BINS_ORDERED, ordered=True)\n"
    "    pivot = sub.pivot(index='quality_bin', columns='method', values='hit_rate').reindex(BINS_ORDERED)\n"
    "    pivot.plot.bar(ax=ax, width=0.7, edgecolor='white')\n"
    "    ax.set_title(f'tau_t = {tau_t}')\n"
    "    ax.set_xlabel('Baseline quality bin')\n"
    "    if ax is axes[0]:\n"
    "        ax.set_ylabel('Hit rate')\n"
    "    ax.tick_params(axis='x', rotation=30)\n"
    "    ax.legend(fontsize=8)\n"
    "\n"
    "plt.suptitle('Hit rate by quality bin', y=1.02)\n"
    "plt.tight_layout()\n"
    "plt.savefig(OUT_DIR / 'hit_rate_by_bin.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
))

cells.append(code(
    "# Score distributions (tanimoto + desirability) per method\n"
    "fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))\n"
    "\n"
    "for method, grp in synth_df.groupby('method'):\n"
    "    if len(grp) > 1:\n"
    "        grp['tanimoto'].plot.kde(ax=axes[0], label=method)\n"
    "        grp['desirability'].plot.kde(ax=axes[1], label=method)\n"
    "\n"
    "axes[0].axvline(TAU_T_LIST[0], color='gray', linestyle='--', linewidth=0.8, label='tau_t')\n"
    "axes[0].set_xlabel('Tanimoto to spec')\n"
    "axes[0].set_title('Substructural conservation')\n"
    "axes[0].legend(fontsize=8)\n"
    "\n"
    "axes[1].axvline(TAU_D, color='gray', linestyle='--', linewidth=0.8, label='tau_d')\n"
    "axes[1].set_xlabel('Desirability')\n"
    "axes[1].set_title('Physicochemical conservation')\n"
    "axes[1].legend(fontsize=8)\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig(OUT_DIR / 'score_distributions.png', dpi=150)\n"
    "plt.show()\n"
))

# Bioactive proximity (diagnostic, not success criterion)
cells.append(code(
    "# Bioactive proximity: Tanimoto(variant, ChEMBL reference) -- diagnostic only.\n"
    "# Not a success criterion; shows how close variants are to known bioactive space.\n"
    "if 'bioactive_proximity' in synth_df.columns:\n"
    "    fig, ax = plt.subplots(figsize=(5, 3))\n"
    "    for method, grp in synth_df.groupby('method'):\n"
    "        grp['bioactive_proximity'].dropna().plot.kde(ax=ax, label=method)\n"
    "    ax.set_xlabel('Tanimoto to ChEMBL reference (diagnostic)')\n"
    "    ax.set_title('Bioactive proximity (diagnostic only)')\n"
    "    ax.legend(fontsize=8)\n"
    "    plt.tight_layout()\n"
    "    plt.savefig(OUT_DIR / 'bioactive_proximity.png', dpi=150)\n"
    "    plt.show()\n"
    "else:\n"
    "    print('bioactive_proximity not in data -- skipping diagnostic plot.')\n"
))

# ─── Stage 5c ─────────────────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Stage 5c — Complementarity (Table 2)\n\n"
    "**Pseudocode**:\n"
    "```\n"
    "for each pair (M_i, M_j):\n"
    "    overlap_ij = |hits_i AND hits_j| / |hits_i OR hits_j|   # Jaccard\n"
    "\n"
    "for each triple (M_i, M_j, M_k):\n"
    "    combined = |hits_i OR hits_j OR hits_k|\n"
    "```\n\n"
    "Low Jaccard = complementary (non-overlapping coverage)."
))

cells.append(code(
    "TAU_T_PILOT = TAU_T_LIST[0]   # most permissive threshold for overlap analysis\n"
    "\n"
    "hit_sets: dict[str, set[str]] = {}\n"
    "for method, grp in synth_df.groupby('method'):\n"
    "    mask = classify_hits(grp, tau_t=TAU_T_PILOT, tau_d=TAU_D)\n"
    "    hit_sets[method] = set(grp.loc[mask, 'variant_smiles'])\n"
    "    print(f'{method:12s}  unique hits: {len(hit_sets[method])}')\n"
    "\n"
    "if len(hit_sets) >= 2:\n"
    "    comp_df = complementarity(hit_sets)\n"
    "    comp_df.to_csv(COMPLEMENT_CSV)\n"
    "    print(f'\\nPairwise Jaccard (tau_t={TAU_T_PILOT}, tau_d={TAU_D}):')\n"
    "    display(comp_df.round(3))\n"
    "else:\n"
    "    print('Need >= 2 methods for complementarity analysis.')\n"
    "    comp_df = pd.DataFrame()\n"
))

cells.append(code(
    "if not comp_df.empty:\n"
    "    fig, ax = plt.subplots(figsize=(max(4, len(hit_sets)), max(3, len(hit_sets))))\n"
    "    sns.heatmap(\n"
    "        comp_df.astype(float), annot=True, fmt='.2f',\n"
    "        cmap='YlOrRd', vmin=0, vmax=1, linewidths=0.5, ax=ax,\n"
    "    )\n"
    "    ax.set_title(f'Pairwise Jaccard overlap (tau_t={TAU_T_PILOT}, tau_d={TAU_D})')\n"
    "    plt.tight_layout()\n"
    "    plt.savefig(OUT_DIR / 'complementarity_heatmap.png', dpi=150)\n"
    "    plt.show()\n"
    "\n"
    "if len(hit_sets) >= 2:\n"
    "    combos = best_combinations(hit_sets, top_k=10)\n"
    "    combos_df = pd.DataFrame([\n"
    "        {'methods': ' + '.join(c), 'unique_hits': n}\n"
    "        for c, n in combos\n"
    "    ])\n"
    "    combos_df.to_csv(COMBOS_CSV, index=False)\n"
    "    print(f'Top combinations (tau_t={TAU_T_PILOT}):')\n"
    "    display(combos_df)\n"
))

# ─── Outputs ──────────────────────────────────────────────────────────────────
cells.append(md("---\n## Outputs"))

cells.append(code(
    "print('=' * 55)\n"
    "print('Pipeline Analysis -- outputs')\n"
    "print('=' * 55)\n"
    "for p in sorted(OUT_DIR.iterdir()):\n"
    "    print(f'  {p.name:<40s} {p.stat().st_size/1e3:6.1f} kB')\n"
))

# ─── Notebook metadata ────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (prexsyn)",
            "language": "python",
            "name": "prexsyn",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "cells": cells,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written : {OUT}")
print(f"Cells   : {len(cells)}")
for i, c in enumerate(cells):
    ctype = c["cell_type"]
    preview = "".join(c["source"])[:60].replace("\n", " ")
    print(f"  [{i:2d}] {ctype:8s}  {preview}")
