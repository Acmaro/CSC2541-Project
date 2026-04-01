"""Insert Table 4 + Figure 8 (saturation analysis) into results_tables.ipynb."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "notebooks" / "results_tables.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

# ── New markdown cell ────────────────────────────────────────────────────────
md_cell = {
    "cell_type": "markdown",
    "id": "sat-exp-md",
    "metadata": {},
    "source": (
        "---\n"
        "## Table 4 / Figure 8: Saturation vs. Expansion\n"
        "\n"
        "**Research question:** Does repeated PrexSyn sampling saturate while modification\n"
        "methods open genuinely new chemical territory?\n"
        "\n"
        "**Table 4** reports, per method:\n"
        "- `Unique Hits` — distinct synthesizable SMILES\n"
        "- `New Territory` — hits absent from the baseline hit set\n"
        "- `% New` — fraction of a method's hits that are novel relative to baseline\n"
        "- `Specs w/ Hit` — how many of the 100 seed specs produced at least one hit\n"
        "- `Hits / Covered Spec` — hit density among specs that produced any hit\n"
        "\n"
        "**Figure 8** shows two panels:\n"
        "- *Left* — per-spec hit-count violin: reveals whether baseline concentrates hits on few specs\n"
        "- *Right* — cumulative discovery curve across specs sorted by baseline quality"
    ),
}

# ── New code cell ────────────────────────────────────────────────────────────
code_src = """\
# ── Table 4: Saturation vs. Expansion ───────────────────────────────────────
synth_col     = PRIMARY_COL
baseline_df   = dfs['Baseline']
baseline_hits = set(baseline_df.loc[baseline_df[synth_col], 'variant_smiles'].dropna())

sat_rows = []
for name in METHODS:
    df = dfs[name]
    if synth_col not in df.columns:
        continue
    hits_df  = df[df[synth_col]].copy()
    unique   = set(hits_df['variant_smiles'].dropna())
    new_hits = unique - baseline_hits if name != 'Baseline' else set()
    pct_new  = round(100 * len(new_hits) / len(unique), 1) if unique else 0.0

    specs_hit = (
        hits_df['spec_smiles'].nunique() if 'spec_smiles' in hits_df.columns else float('nan')
    )
    mean_per = round(len(unique) / specs_hit, 1) if specs_hit > 0 else 0.0

    sat_rows.append({
        'Method':              METHOD_LABELS[name],
        'Total Variants':      len(df),
        'Unique Hits':         len(unique),
        'New Territory':       len(new_hits) if name != 'Baseline' else '\\u2014',
        '% New':               f'{pct_new:.1f}%' if name != 'Baseline' else '\\u2014',
        'Specs w/ Hit':        specs_hit,
        'Hits / Covered Spec': mean_per,
    })

df_sat = pd.DataFrame(sat_rows)
print(f'Table 4: Saturation vs. Expansion  (primary depth: {PRIMARY_DEPTH})')
print()
print(df_sat.to_string(index=False))

# ── Figure 8 ─────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: per-spec hit count violin
per_spec_vals, vlabels = [], []
all_specs = dfs['Baseline']['spec_smiles'].unique() if 'spec_smiles' in dfs['Baseline'].columns else []
for name in METHODS:
    df = dfs[name]
    if synth_col not in df.columns or 'spec_smiles' not in df.columns:
        continue
    counts = (
        df[df[synth_col]]
        .groupby('spec_smiles')['variant_smiles']
        .nunique()
        .reindex(all_specs, fill_value=0)
        .values
    )
    per_spec_vals.append(counts)
    vlabels.append(name)

parts = axes[0].violinplot(per_spec_vals, showmedians=True, showextrema=True)
for pc in parts['bodies']:
    pc.set_alpha(0.7)
axes[0].set_xticks(range(1, len(vlabels) + 1))
axes[0].set_xticklabels(vlabels, rotation=20, ha='right', fontsize=9)
axes[0].set_ylabel('Unique synthesizable hits per spec')
axes[0].set_title('Per-spec Hit Count Distribution', fontweight='bold')
axes[0].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# Right: cumulative discovery curve sorted by baseline quality
bq_map = None
if 'baseline_quality' in baseline_df.columns:
    bq_map = (
        baseline_df[['spec_smiles', 'baseline_quality']]
        .drop_duplicates()
        .set_index('spec_smiles')['baseline_quality']
    )

colors_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
for ci, name in enumerate(METHODS):
    df = dfs[name]
    if synth_col not in df.columns or 'spec_smiles' not in df.columns:
        continue
    per_spec = (
        df[df[synth_col]]
        .groupby('spec_smiles')['variant_smiles']
        .nunique()
        .reset_index(name='n_hits')
    )
    if bq_map is not None:
        per_spec['bq'] = per_spec['spec_smiles'].map(bq_map)
        per_spec = per_spec.sort_values('bq')
    cum = per_spec['n_hits'].cumsum().values
    lw = 2.5 if name == 'Baseline' else 1.8
    ls = '--' if name == 'Baseline' else '-'
    axes[1].plot(
        range(1, len(cum) + 1), cum,
        label=name, linewidth=lw, linestyle=ls,
        color=colors_cycle[ci % len(colors_cycle)],
    )

axes[1].set_xlabel('Specs ranked by baseline quality (low \\u2192 high)')
axes[1].set_ylabel('Cumulative unique synthesizable hits')
axes[1].set_title('Hit Discovery Curve\\n(sorted by spec difficulty)', fontweight='bold')
axes[1].legend(fontsize=9, frameon=False)

fig.suptitle(
    f'Figure 8: Saturation vs. Expansion  (AiZynthFinder depth {PRIMARY_DEPTH})',
    fontweight='bold', y=1.02,
)
plt.tight_layout()
out_path = GEN_DIR / 'fig8_saturation_expansion.png'
plt.savefig(out_path, bbox_inches='tight', dpi=150)
plt.show()
print(f'Saved -> {out_path}')
"""

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "sat-exp-code",
    "metadata": {},
    "outputs": [],
    "source": code_src,
}

# Insert after cell index 9 (Table 3 code), before cell 10 (LaTeX Output header)
nb["cells"].insert(10, code_cell)
nb["cells"].insert(10, md_cell)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done. Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    tag = c.get("id", "")
    src = "".join(c["source"])[:70].replace("\n", " ")
    print(f"  [{i}] {c['cell_type']:<8} {tag:<20} {src}")
