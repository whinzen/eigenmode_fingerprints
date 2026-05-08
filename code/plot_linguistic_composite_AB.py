from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
PANG = Path.home() / "eigenmode_fingerprints" / "pang_out"

OUT = PANG / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = PANG / "word_transition_geometry" / "word_transition_summary.csv"
FULL_CSV = PANG / "word_transition_geometry" / "word_transition_geometry.csv"

METRICS = [
    ("shift", "Shift"),
    ("pred_error_ar", "AR error"),
    ("pred_error_subspace", "Subspace exit"),
    ("curvature", "Curvature"),
]


def pooled_sd(mean1, sem1, n1, mean2, sem2, n2):
    sd1 = sem1 * np.sqrt(n1)
    sd2 = sem2 * np.sqrt(n2)
    if n1 < 2 or n2 < 2:
        return np.nan
    return np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))


def cohens_d_from_summary(mean_within, sem_within, n_within,
                          mean_boundary, sem_boundary, n_boundary):
    sp = pooled_sd(mean_within, sem_within, n_within,
                   mean_boundary, sem_boundary, n_boundary)
    if not np.isfinite(sp) or sp == 0:
        return np.nan
    return (mean_boundary - mean_within) / sp


def significance_stars(p):
    if p < 1e-10:
        return "***"
    if p < 1e-5:
        return "**"
    if p < 0.01:
        return "*"
    return "n.s."


summary_df = pd.read_csv(SUMMARY_CSV)
full_df = pd.read_csv(FULL_CSV)

labels = []
means_within, means_boundary = [], []
sems_within, sems_boundary = [], []
cohens_ds, pvals = [], []

for metric, label in METRICS:
    row = summary_df.loc[summary_df["measure"] == metric].iloc[0]

    labels.append(label)
    means_within.append(row["within_mean"])
    means_boundary.append(row["boundary_mean"])
    sems_within.append(row["within_sem"])
    sems_boundary.append(row["boundary_sem"])
    pvals.append(row["p"])

    cohens_ds.append(
        cohens_d_from_summary(
            row["within_mean"], row["within_sem"], row["n_within"],
            row["boundary_mean"], row["boundary_sem"], row["n_boundary"],
        )
    )

means_within = np.asarray(means_within, float)
means_boundary = np.asarray(means_boundary, float)
sems_within = np.asarray(sems_within, float)
sems_boundary = np.asarray(sems_boundary, float)
cohens_ds = np.asarray(cohens_ds, float)
pvals = np.asarray(pvals, float)

corr_metrics = [m for m, _ in METRICS]
corr_labels = [lab for _, lab in METRICS]
corr_df = full_df[corr_metrics].dropna().copy()

# Pearson correlations
corr = np.corrcoef(corr_df.values.T)

fig, axes = plt.subplots(
    1, 2,
    figsize=(11.8, 4.8),
    gridspec_kw={"width_ratios": [1.35, 1.0]},
)

# ---------- Panel A ----------
ax = axes[0]
x = np.arange(len(labels))
width = 0.34

within_color = "0.78"
boundary_color = "0.35"

ax.bar(
    x - width / 2,
    means_within,
    width,
    yerr=sems_within,
    capsize=4,
    label="Within sentence",
    color=within_color,
    edgecolor="none",
    error_kw={"elinewidth": 1.1, "ecolor": "0.2"},
)
ax.bar(
    x + width / 2,
    means_boundary,
    width,
    yerr=sems_boundary,
    capsize=4,
    label="Boundary",
    color=boundary_color,
    edgecolor="none",
    error_kw={"elinewidth": 1.1, "ecolor": "0.2"},
)

y_max = np.nanmax(np.concatenate([
    means_within + sems_within,
    means_boundary + sems_boundary,
]))
y_min = np.nanmin(np.concatenate([
    means_within - sems_within,
    means_boundary - sems_boundary,
]))

y_top = y_max * 1.35 if y_max > 0 else y_max + 1.0
y_bottom = min(0.0, y_min * 1.10 if y_min < 0 else 0.0)
ax.set_ylim(y_bottom, y_top)

for i, (mw, mb, d, p) in enumerate(zip(means_within, means_boundary, cohens_ds, pvals)):
    y = max(mw + sems_within[i], mb + sems_boundary[i])
    ax.text(
        i,
        y + 0.04 * (y_top - y_bottom),
        f"d={d:.2f}\n{significance_stars(p)}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Metric value")
ax.set_title("A. Boundary effects in transition metrics", pad=14)
ax.legend(frameon=False, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ---------- Panel B ----------
ax = axes[1]

# Use the true observed correlation range, but keep upper bound at 1.
vmin = np.floor(np.nanmin(corr) * 10) / 10
vmax = 1.0
norm = plt.Normalize(vmin=vmin, vmax=vmax)

im = ax.imshow(
    corr,
    vmin=vmin,
    vmax=vmax,
    cmap="RdBu_r",
)

ax.set_xticks(range(len(corr_metrics)))
ax.set_yticks(range(len(corr_metrics)))
ax.set_xticklabels(corr_labels, rotation=35, ha="right")
ax.set_yticklabels(corr_labels)
ax.set_title("B. Pearson correlations among metrics", pad=14)

for i in range(len(corr_metrics)):
    for j in range(len(corr_metrics)):
        val = corr[i, j]
        color = "white" if norm(val) > 0.72 or norm(val) < 0.25 else "black"
        ax.text(
            j,
            i,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            color=color,
        )

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson r", fontsize=9)
cbar.ax.tick_params(labelsize=8)
cbar.outline.set_visible(False)

fig.suptitle(
    "Boundary-related increases and representational correlations among token-level transition metrics",
    y=1.03,
    fontsize=13,
)

fig.tight_layout()

png = OUT / "figure_transition_metrics_boundary_and_correlations.png"
pdf = OUT / "figure_transition_metrics_boundary_and_correlations.pdf"
svg = OUT / "figure_transition_metrics_boundary_and_correlations.svg"

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")
plt.close(fig)

print(f"✅ wrote {png}")
print(f"✅ wrote {pdf}")
print(f"✅ wrote {svg}")
print("Panel A error bars: SEM")
print("Panel B correlations: Pearson")