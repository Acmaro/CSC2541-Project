import json, sys
from pathlib import Path

NB_PATH   = Path(r"D:\AI4DD Project\CSC2541-Project\notebooks\results_tables.ipynb")
TARGET_ID = '5a7cc3a810'

# ── Cell sources ──────────────────────────────────────────────────────────────

FIG10_MD = """---
## Figure 10: Diversity Saturation Curve

**Research question:** Does repeated PrexSyn sampling saturate while modification methods keep discovering new synthesizable hits?

- *Left* — Cumulative unique synthesizable hits vs. variants sampled (mean ± 1 SD over 20 random orderings)
- *Right* — Marginal discovery rate: new unique hits per 100 variants sampled (smoothed). Rate approaching 0 confirms the method has exhausted its accessible hit space.

PrexSyn resampling draws from a small, discrete latent neighbourhood per seed. Modification methods apply combinatorial transformations not bounded by that ceiling."""

FIG10_CODE = r"""# -- Figure 10: Diversity Saturation Curve ---------------------------------------
# Left:  cumulative unique synthesizable hits (mean +/- 1 SD, 20 random orderings)
# Right: marginal discovery rate (new hits per 100 variants, smoothed)

N_PERMS_FIG10 = 20
STEP_FIG10    = 50
SMOOTH_W      = 5
rng_fig10     = np.random.default_rng(0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
all_curves_fig10 = {}

for name in VIZ_METHODS:
    df     = viz_dfs[name]
    col    = VIZ_COL if VIZ_COL in df.columns else 'is_synth'
    is_hit = df[col].fillna(False).values.astype(bool)
    smiles = df['variant_smiles'].values
    n      = len(df)

    curves = []
    for _ in range(N_PERMS_FIG10):
        perm = rng_fig10.permutation(n)
        seen = set()
        pts  = []
        for j, idx in enumerate(perm):
            if is_hit[idx]:
                seen.add(smiles[idx])
            if (j + 1) % STEP_FIG10 == 0:
                pts.append(len(seen))
        if pts:
            curves.append(pts)

    if not curves:
        continue

    min_len = min(len(c) for c in curves)
    arr     = np.array([c[:min_len] for c in curves])
    xs_m    = np.arange(1, min_len + 1) * STEP_FIG10
    mean_m  = arr.mean(axis=0)
    std_m   = arr.std(axis=0)
    all_curves_fig10[name] = (xs_m, mean_m, std_m)

    c   = PALETTE[name]
    lw  = 2.5 if name == 'Baseline' else 1.8
    ls  = '--' if name == 'Baseline' else '-'

    axes[0].plot(xs_m, mean_m, lw=lw, ls=ls, color=c, label=name, zorder=3)
    axes[0].fill_between(xs_m, mean_m - std_m, mean_m + std_m, alpha=0.13, color=c)

    if len(mean_m) > 1:
        incr   = np.diff(mean_m) / STEP_FIG10 * 100
        kernel = np.ones(SMOOTH_W) / SMOOTH_W
        smooth = np.convolve(incr, kernel, mode='same')
        axes[1].plot(xs_m[1:], smooth, lw=lw, ls=ls, color=c, label=name, zorder=3)

# Annotate baseline saturation point
if 'Baseline' in all_curves_fig10:
    xs_b, mean_b, std_b = all_curves_fig10['Baseline']
    incr_b  = np.diff(mean_b) / STEP_FIG10 * 100
    sat_idx = next((i for i, v in enumerate(incr_b) if v < 0.5), None)
    if sat_idx is not None:
        sat_x = int(xs_b[sat_idx])
        for _ax in axes:
            _ax.axvline(sat_x, color=PALETTE['Baseline'], lw=1.2, ls=':', alpha=0.8)
        axes[0].text(
            sat_x + STEP_FIG10 * 1.5, mean_b[sat_idx] * 0.5,
            f'Baseline plateau\n({sat_x:,} variants)',
            color=PALETTE['Baseline'], fontsize=8, va='center',
        )

axes[0].set_xlabel('Variants sampled (averaged over 20 random orderings)')
axes[0].set_ylabel('Cumulative unique synthesizable hits')
axes[0].set_title('Cumulative Hit Discovery\n(flattening = sample space exhausted)', fontweight='bold')
axes[0].legend(fontsize=9, frameon=False)
axes[0].grid(True, alpha=0.2)

axes[1].set_xlabel('Variants sampled')
axes[1].set_ylabel('New unique hits per 100 variants (smoothed)')
axes[1].set_title('Marginal Discovery Rate\n(rate approaching 0 confirms saturation)', fontweight='bold')
axes[1].axhline(0.5, color='grey', lw=1, ls=':', alpha=0.6, label='Saturation threshold (0.5)')
axes[1].legend(fontsize=9, frameon=False)
axes[1].grid(True, alpha=0.2)
axes[1].set_ylim(bottom=0)

fig.suptitle(
    'Figure 10: Diversity Saturation Curve -- PrexSyn Plateaus, Modification Methods Expand',
    fontweight='bold', y=1.02,
)
plt.tight_layout()
out_path = GEN_DIR / 'fig10_diversity_saturation.png'
plt.savefig(out_path, bbox_inches='tight', dpi=150)
plt.show()
print(f'Saved -> {out_path}')
"""

FIG11_MD = """---
## Figure 11: Method Complementarity Heatmap

**Research question:** Do modification methods discover overlapping or complementary sets of synthesizable hits?

- *Left* — 4×4 Jaccard similarity heatmap between hit sets of the four modification methods. Lower Jaccard = more complementary (the two methods find different molecules).
- *Right* — Total hits vs. exclusive hits (found only by that method and not by any other). High exclusive fraction = strong unique contribution."""

FIG11_CODE = r"""# -- Figure 11: Method Complementarity Heatmap ----------------------------------
MOD_METHODS_11 = ['CReM', 'mmpdb', 'JT-VAE', 'LibINVENT']

# Build hit sets per method
hit_sets_11 = {}
for name in MOD_METHODS_11:
    df  = viz_dfs[name]
    col = VIZ_COL if VIZ_COL in df.columns else 'is_synth'
    hit_sets_11[name] = set(
        df.loc[df[col].fillna(False).astype(bool), 'variant_smiles'].dropna()
    )

# Jaccard matrix
n_mod       = len(MOD_METHODS_11)
jac_mat     = np.zeros((n_mod, n_mod))
overlap_mat = np.zeros((n_mod, n_mod), dtype=int)
for i, a in enumerate(MOD_METHODS_11):
    for j, b in enumerate(MOD_METHODS_11):
        inter             = len(hit_sets_11[a] & hit_sets_11[b])
        union             = len(hit_sets_11[a] | hit_sets_11[b])
        jac_mat[i, j]     = inter / union if union > 0 else 0.0
        overlap_mat[i, j] = inter

off_diag = jac_mat[~np.eye(n_mod, dtype=bool)]
vmax_jac  = max(float(off_diag.max()) * 1.5, 0.05) if len(off_diag) > 0 else 0.3

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: heatmap
im = axes[0].imshow(jac_mat, vmin=0, vmax=vmax_jac, cmap='YlOrRd', aspect='auto')
plt.colorbar(im, ax=axes[0], label='Jaccard index')
axes[0].set_xticks(range(n_mod))
axes[0].set_yticks(range(n_mod))
axes[0].set_xticklabels(MOD_METHODS_11, rotation=30, ha='right', fontsize=10)
axes[0].set_yticklabels(MOD_METHODS_11, fontsize=10)
for i in range(n_mod):
    for j in range(n_mod):
        v   = jac_mat[i, j]
        cnt = overlap_mat[i, j]
        lbl = f'{v:.3f}\n({cnt})' if i != j else f'{v:.3f}\n(self)'
        fc  = 'white' if v > vmax_jac * 0.6 else 'black'
        axes[0].text(j, i, lbl, ha='center', va='center', fontsize=8, color=fc)
axes[0].set_title(
    'Pairwise Jaccard Overlap (hit sets)\nLower = more complementary',
    fontweight='bold',
)

# Right: total vs exclusive hits
all_mod_set = set().union(*[hit_sets_11[m] for m in MOD_METHODS_11])
excl = {
    name: len(hit_sets_11[name] - set().union(
        *[hit_sets_11[m] for m in MOD_METHODS_11 if m != name]
    ))
    for name in MOD_METHODS_11
}

x_pos      = range(n_mod)
total_hits  = [len(hit_sets_11[m]) for m in MOD_METHODS_11]
excl_hits   = [excl[m]             for m in MOD_METHODS_11]
colors_mod  = [PALETTE[m]          for m in MOD_METHODS_11]
y_max       = max(total_hits) if total_hits else 1

axes[1].bar(x_pos, total_hits, color=colors_mod, width=0.5, alpha=0.45, label='Total hits')
axes[1].bar(x_pos, excl_hits,  color=colors_mod, width=0.5, alpha=1.0,
            label='Exclusive hits (unique to this method)')
for i, (t, e) in enumerate(zip(total_hits, excl_hits)):
    pct = round(e / t * 100) if t > 0 else 0
    axes[1].text(i, t + y_max * 0.025,
                 f'{e}/{t}\n({pct}% excl.)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

axes[1].set_xticks(list(x_pos))
axes[1].set_xticklabels(MOD_METHODS_11, rotation=20, ha='right', fontsize=9)
axes[1].set_ylabel('Unique synthesizable hits')
axes[1].set_title(
    'Total vs. Exclusive Hits per Method\nExclusive = not found by any other method',
    fontweight='bold',
)
axes[1].legend(fontsize=9, frameon=False)
axes[1].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
axes[1].set_ylim(0, y_max * 1.20)

fig.suptitle(
    'Figure 11: Method Complementarity -- Modification Methods Cover Non-overlapping Territory',
    fontweight='bold', y=1.02,
)
plt.tight_layout()
out_path = GEN_DIR / 'fig11_complementarity_heatmap.png'
plt.savefig(out_path, bbox_inches='tight', dpi=150)
plt.show()

print('Jaccard overlap (off-diagonal pairs):')
for i, a in enumerate(MOD_METHODS_11):
    for j, b in enumerate(MOD_METHODS_11):
        if j > i:
            print(f'  {a} x {b}: J = {jac_mat[i,j]:.4f}  ({overlap_mat[i,j]} shared hits)')
print(f'\nUnion of all modification method hits: {len(all_mod_set):,}')
print(f'Saved -> {out_path}')
"""

FIG12_MD = """---
## Figure 12: Property Conservation Scatter Plot

**Research question:** In the (substructural conservation, physicochemical desirability) plane, where do synthesizable hits from each method cluster?

- Rule-based methods (CReM, mmpdb) apply local graph edits: expected to cluster top-right (high Tanimoto, high desirability).
- ML methods (JT-VAE, LibINVENT) make larger structural moves: expected to spread left (lower Tanimoto, still drug-like), revealing their reach into structurally distant but valid chemical space.

Dashed lines show evaluation thresholds τ_t (substructural conservation) and τ_d (desirability)."""

FIG12_CODE = r"""# -- Figure 12: Property Conservation Scatter Plot ------------------------------
fig, ax = plt.subplots(figsize=(8, 7))

for name in VIZ_METHODS:
    df   = viz_dfs[name]
    col  = VIZ_COL if VIZ_COL in df.columns else 'is_synth'
    hits = df[df[col].fillna(False).astype(bool)].copy()
    if hits.empty:
        continue
    ax.scatter(
        hits['tanimoto'].clip(0, 1),
        hits['desirability'].clip(0, 1),
        color=PALETTE[name],
        label=f'{name}  (n={len(hits):,})',
        s=18, alpha=0.40, linewidths=0, zorder=2,
    )

# Decision thresholds
ax.axvline(TAU_T_VIZ, color='#444444', lw=1.2, ls='--', alpha=0.65,
           label=f'tau_t = {TAU_T_VIZ}  (Tanimoto threshold)')
ax.axhline(TAU_D_VIZ, color='#444444', lw=1.2, ls=':',  alpha=0.65,
           label=f'tau_d = {TAU_D_VIZ}  (Desirability threshold)')

# Quadrant labels
kw = dict(transform=ax.transAxes, fontsize=8, color='#999999')
ax.text(0.02, 0.98, 'Diverse & drug-like\n(low sim., high des.)',  va='top',    ha='left',  **kw)
ax.text(0.98, 0.98, 'Close analogs & drug-like\n(high sim., high des.)', va='top', ha='right', **kw)
ax.text(0.02, 0.02, 'Off-target / rejected',                         va='bottom', ha='left',  **kw)

ax.set_xlabel('Substructural conservation -- ECFP4 Tanimoto to reference (spec)', fontsize=11)
ax.set_ylabel('Physicochemical desirability (geometric mean)', fontsize=11)
ax.set_title(
    'Figure 12: Property Conservation of Synthesizable Hits\n'
    'Rule-based methods conserve structure; ML methods explore broader space',
    fontweight='bold',
)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=9, frameon=True, framealpha=0.85, loc='upper left')
ax.grid(True, alpha=0.15)

plt.tight_layout()
out_path = GEN_DIR / 'fig12_property_conservation.png'
plt.savefig(out_path, bbox_inches='tight', dpi=150)
plt.show()
print(f'Saved -> {out_path}')
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_md(src):
    lines = src.lstrip('\n').splitlines(keepends=True)
    if lines and lines[-1].endswith('\n'):
        lines[-1] = lines[-1].rstrip('\n')
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def make_code(src):
    lines = src.lstrip('\n').splitlines(keepends=True)
    if lines and lines[-1].endswith('\n'):
        lines[-1] = lines[-1].rstrip('\n')
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }

# ── Modify notebook ───────────────────────────────────────────────────────────

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
idx = next((i for i, c in enumerate(cells) if c.get('id') == TARGET_ID), None)
if idx is None:
    print(f"ERROR: cell id='{TARGET_ID}' not found", file=sys.stderr)
    sys.exit(1)

new_cells = [
    make_md(FIG10_MD),
    make_code(FIG10_CODE),
    make_md(FIG11_MD),
    make_code(FIG11_CODE),
    make_md(FIG12_MD),
    make_code(FIG12_CODE),
]

cells[idx + 1:idx + 1] = new_cells
print(f"Inserted 6 cells after cell idx={idx} (id={TARGET_ID})")
print(f"Notebook now has {len(cells)} cells")

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Saved.")
