#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "github_release/precomputed/geometry_specificity"
OUT = BASE / "github_release/figures"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

FILE_A = ROOT / "cortical_vs_spherical_basis_R2.csv"

FILE_B = (
    ROOT / "onset_comparison/"
    "participant_onset_LB_vs_LOPO_PCA_bilateral.csv"
)

FILE_C = (
    ROOT / "folded_vs_sphere_functional_alignment/"
    "group_PCA_folded_vs_sphere.csv"
)

FILE_D_FEM = ROOT / "heldout_variance/variance_capture_summary.csv"

FILE_D_EUC = (
    ROOT / "heldout_variance_euclidean/"
    "variance_capture_summary.csv"
)

for f in [FILE_A, FILE_B, FILE_C, FILE_D_FEM, FILE_D_EUC]:
    if not f.exists():
        raise FileNotFoundError(f)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

a = pd.read_csv(FILE_A)
b = pd.read_csv(FILE_B)
c = pd.read_csv(FILE_C)
d_fem = pd.read_csv(FILE_D_FEM).sort_values("K")
d_euc = pd.read_csv(FILE_D_EUC).sort_values("K")

# Bilateral mean for panel A
a_group = (
    a.groupby("K")[["R2_cortical_LB", "R2_spherical_LB"]]
    .mean()
    .reset_index()
)

# Participant mean and SEM for panel B
b_group = (
    b.groupby("K")[["R2_LB", "R2_PCA"]]
    .agg(["mean", "sem"])
)

# ------------------------------------------------------------
# Plotting style
# ------------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FOLDED = "#245B8E"
SPHERE = "#C16C37"
PCA = "#338A75"
EUC = "#9A669B"

fig, axes = plt.subplots(
    2, 2,
    figsize=(11.4, 8.3),
    constrained_layout=True
)

axA, axB = axes[0]
axC, axD = axes[1]

# ============================================================
# A. Folded versus spherical reconstruction
# ============================================================

axA.plot(
    a_group["K"],
    a_group["R2_cortical_LB"],
    "o-",
    color=FOLDED,
    lw=2,
    ms=4,
    label="Folded cortex"
)

axA.plot(
    a_group["K"],
    a_group["R2_spherical_LB"],
    "s-",
    color=SPHERE,
    lw=2,
    ms=4,
    label="Sphere"
)

axA.set(
    xlabel="Number of eigenmodes (K)",
    ylabel="Mass-weighted reconstruction R²",
    title="Sentence-onset response reconstruction"
)

axA.legend(frameon=False)
axA.set_ylim(0, 0.9)

# ============================================================
# B. Functional PCA versus folded eigenmodes
# ============================================================

K_B = b_group.index.to_numpy()

for column, label, color, marker in [
    ("R2_LB", "Folded eigenmodes", FOLDED, "o"),
    ("R2_PCA", "Functional PCA", PCA, "s"),
]:
    mean = b_group[(column, "mean")].to_numpy()
    sem = b_group[(column, "sem")].to_numpy()

    axB.errorbar(
        K_B, mean,
        yerr=sem,
        fmt=marker + "-",
        color=color,
        lw=2,
        ms=4,
        capsize=2,
        label=label
    )

axB.axvline(20, color="0.6", ls="--", lw=1)

axB.set(
    xlabel="Number of components (K)",
    ylabel="Held-out reconstruction R²",
    title="Functional versus geometric compression"
)

axB.legend(frameon=False)
axB.set_ylim(0, 0.75)

# ============================================================
# C. Functional-geometric subspace alignment
# ============================================================

axC.plot(
    c["K"],
    c["folded_overlap_mean"],
    "o-",
    color=FOLDED,
    lw=2,
    ms=4,
    label="Functional PCA – folded"
)

axC.plot(
    c["K"],
    c["sphere_overlap_mean"],
    "s-",
    color=SPHERE,
    lw=2,
    ms=4,
    label="Functional PCA – sphere"
)

axC.axvline(20, color="0.6", ls="--", lw=1)

axC.set(
    xlabel="Subspace dimensionality (K)",
    ylabel="Principal-angle subspace overlap",
    title="Functional–geometric alignment"
)

axC.legend(frameon=False)
axC.set_ylim(0.25, 0.70)

# ============================================================
# D. Direct folded-minus-spherical variance capture
# ============================================================

axD.axhline(0, color="0.55", ls="--", lw=1)

for df, label, color, marker in [
    (d_fem, "FEM surface-area weighting", FOLDED, "o"),
    (d_euc, "Uniform vertex weighting", EUC, "s"),
]:
    K = df["K"].to_numpy()
    delta = df["delta_pp"].to_numpy()
    low = df["ci_low_pp"].to_numpy()
    high = df["ci_high_pp"].to_numpy()

    axD.errorbar(
        K,
        delta,
        yerr=np.vstack([delta - low, high - delta]),
        fmt=marker + "-",
        color=color,
        lw=2,
        ms=4,
        capsize=3,
        label=label
    )

axD.set(
    xlabel="Number of eigenmodes (K)",
    ylabel="Folded − spherical variance capture (pp)",
    title="Direct functional variance capture"
)

axD.set_xticks(d_fem["K"])
axD.set_ylim(-0.05, 1.3)
axD.legend(frameon=False)

# ------------------------------------------------------------
# Panel labels
# ------------------------------------------------------------

for label, ax in zip("ABCD", [axA, axB, axC, axD]):
    ax.text(
        -0.12, 1.08,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left"
    )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

pdf = OUT / "Figure_S3_geometry_specificity_FINAL.pdf"
png = OUT / "Figure_S3_geometry_specificity_FINAL.png"

fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, dpi=300, bbox_inches="tight")

plt.close(fig)

print("Saved:")
print(pdf)
print(png)

# ------------------------------------------------------------
# Numerical checks
# ------------------------------------------------------------

k24_fem = d_fem.loc[d_fem["K"] == 24].iloc[0]
k24_euc = d_euc.loc[d_euc["K"] == 24].iloc[0]

assert np.isclose(k24_fem["delta_pp"], 0.882907, atol=1e-4)
assert np.isclose(k24_euc["delta_pp"], 0.420945, atol=1e-4)

print("\nK=24 checks passed:")
print("FEM difference:", k24_fem["delta_pp"])
print("Euclidean difference:", k24_euc["delta_pp"])
