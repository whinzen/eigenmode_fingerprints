#!/usr/bin/env python
"""
Integrated sentence-onset cortical/eigenmode figure.

A  Vertex-space structured-null validation
B  Empirical -> K20 reconstruction -> residual
C  Sentence-onset eigenmode profile
D  Preferential low-mode concentration relative to empirical cortex

All analyses use the definitive nuisance-corrected sentence-onset result.

The residual in B is shown as a smaller inset but uses exactly the same
colour scale as empirical and K20 maps.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from nilearn import datasets, plotting


# ============================================================
# Paths
# ============================================================

BASE = Path.home() / "eigenmode_fingerprints"

VROOT = (
    BASE / "pang_out" / "nuisance_robustness"
    / "vertex_space_linguistic_encoding"
)

QC = VROOT / "structured_null_qc"

RECON = (
    VROOT / "sentence_onset_K20_reconstruction"
)

NULLDIR = (
    VROOT / "onset_vertexwise_null_10000"
)

OUT = (
    BASE / "pang_out" / "paper_figures"
    / "sentence_onset_integrated"
)
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load definitive vertex-space results
# ============================================================

emp = {
    h: np.load(
        RECON / f"sentence_onset_empirical_demeaned_{h}.npy"
    )
    for h in ["L", "R"]
}

recon = {
    h: np.load(
        RECON / f"sentence_onset_K20_reconstruction_{h}.npy"
    )
    for h in ["L", "R"]
}

resid = {
    h: np.load(
        RECON / f"sentence_onset_K20_residual_{h}.npy"
    )
    for h in ["L", "R"]
}

valid = {
    h: np.load(
        VROOT / f"valid_vertex_mask_{h}.npy"
    ).astype(bool)
    for h in ["L", "R"]
}

recon_summary = pd.read_csv(
    RECON / "sentence_onset_K20_reconstruction_summary.csv"
)

r2L = float(
    recon_summary.loc[
        recon_summary.hemisphere == "L",
        "R2_K20"
    ].iloc[0]
)

r2R = float(
    recon_summary.loc[
        recon_summary.hemisphere == "R",
        "R2_K20"
    ].iloc[0]
)


# ============================================================
# A: definitive 10,000-permutation vertex-space null
# ============================================================

rms_files = sorted(
    NULLDIR.glob("rms_*.csv")
)

assert len(rms_files) == 100

rms_null = pd.concat(
    [pd.read_csv(f) for f in rms_files],
    ignore_index=True
).sort_values("perm")

assert len(rms_null) == 10000

null_rms = rms_null["group_map_rms"].to_numpy()

EMP_RMS = 0.007737

n_ge = int(
    np.sum(null_rms >= EMP_RMS)
)

p_rms = (
    n_ge + 1
) / (
    len(null_rms) + 1
)

print("\nVertex-space validation")
print("Empirical RMS:", EMP_RMS)
print("Null mean:", null_rms.mean())
print("n >= empirical:", n_ge)
print("p:", p_rms)

# End-to-end provenance check
assert n_ge == 359
assert abs(p_rms - 0.035996400359964) < 1e-12


# ============================================================
# Surface setup
# ============================================================

fs = datasets.fetch_surf_fsaverage(
    mesh="fsaverage5"
)

surf = {
    "L": fs.infl_left,
    "R": fs.infl_right,
}

sulc = {
    "L": fs.sulc_left,
    "R": fs.sulc_right,
}


# Common scale for empirical, reconstruction AND residual.
all_emp = np.concatenate([
    emp["L"][valid["L"]],
    emp["R"][valid["R"]],
])

VMAX = np.nanpercentile(
    np.abs(all_emp),
    99
)

VMIN = -VMAX

norm = mpl.colors.TwoSlopeNorm(
    vmin=VMIN,
    vcenter=0,
    vmax=VMAX,
)

CMAP = "RdBu_r"


# ============================================================
# Locate corrected eigenmode/concentration data
# ============================================================

FIG3 = (
    BASE / "pang_out" / "nuisance_robustness"
    / "figure3_corrected"
)

print("\nSearching corrected Figure-3 products...")

if FIG3.exists():
    for p in sorted(FIG3.rglob("*.csv")):
        if (
            "concentration" in p.name.lower()
            or "k20" in p.name.lower()
            or "subject" in p.name.lower()
        ):
            print(" ", p)


# ============================================================
# Helpers
# ============================================================

def add_brain_axis(
    fig,
    rect,
    data,
    hemi,
    view="lateral",
):
    """
    Add one Nilearn surface brain into arbitrary figure coordinates.
    rect = [left,bottom,width,height]
    """

    ax = fig.add_axes(
        rect,
        projection="3d"
    )

    plotting.plot_surf_stat_map(
        surf[hemi],
        data,
        hemi=(
            "left"
            if hemi == "L"
            else "right"
        ),
        view=view,

        bg_map=sulc[hemi],
        bg_on_data=True,
        darkness=0.65,

        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        symmetric_cbar=True,

        threshold=None,
        colorbar=False,

        axes=ax,
    )

    return ax


# ============================================================
# Main figure
# ============================================================

fig = plt.figure(
    figsize=(15.5, 9.2),
    facecolor="white",
)

gs = GridSpec(
    2,
    2,
    figure=fig,
    left=0.07,
    right=0.97,
    bottom=0.09,
    top=0.95,
    width_ratios=[1.12, 1.0],
    height_ratios=[1.0, 1.0],
    wspace=0.25,
    hspace=0.30,
)


# ============================================================
# PANEL A
# Vertex-space validation
# ============================================================

axA = fig.add_subplot(
    gs[0, 0]
)

bins = np.linspace(
    null_rms.min(),
    null_rms.max(),
    42,
)

axA.hist(
    null_rms,
    bins=bins,
    density=True,
    edgecolor="none",
    alpha=0.75,
)

axA.axvline(
    EMP_RMS,
    linewidth=2.2,
)

axA.text(
    EMP_RMS,
    axA.get_ylim()[1] * 0.88,
    " empirical",
    rotation=90,
    va="top",
    ha="left",
    fontsize=10,
)

axA.set_xlabel(
    r"Whole-cortex group-map RMS of $\beta^*$"
)

axA.set_ylabel(
    "Null density"
)

axA.set_title(
    "A   Sentence-onset response exceeds\n"
    "the structured temporal null",
    loc="left",
    fontweight="bold",
    fontsize=12,
)

axA.text(
    0.97,
    0.94,
    (
        rf"$p_{{\rm perm}}={p_rms:.3f}$"
        "\n"
        r"$N_{\rm perm}=10{,}000$"
    ),
    transform=axA.transAxes,
    ha="right",
    va="top",
    fontsize=10,
)

axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)


# ============================================================
# PANEL B container
# ============================================================

axB = fig.add_subplot(
    gs[0, 1]
)

axB.axis("off")

axB.set_title(
    "B   Low-order eigenmodes preserve broad spatial structure",
    loc="left",
    fontweight="bold",
    fontsize=12,
    pad=8,
)


# Get B panel bounding box in figure coordinates
bbox = axB.get_position()

x0 = bbox.x0
y0 = bbox.y0
w = bbox.width
h = bbox.height


# ------------------------------------------------------------
# B layout:
#
#       EMPIRICAL       K20
#       L      R       L      R
#
#                    residual inset
#
# Residual is deliberately smaller, but SAME colour scale.
# ------------------------------------------------------------

brain_y = y0 + 0.36 * h
brain_h = 0.48 * h

bw = 0.205 * w
gap = 0.015 * w


# Empirical L/R
add_brain_axis(
    fig,
    [x0 + 0.00*w, brain_y, bw, brain_h],
    emp["L"],
    "L",
)

add_brain_axis(
    fig,
    [x0 + 0.20*w, brain_y, bw, brain_h],
    emp["R"],
    "R",
)


# K20 L/R
add_brain_axis(
    fig,
    [x0 + 0.48*w, brain_y, bw, brain_h],
    recon["L"],
    "L",
)

add_brain_axis(
    fig,
    [x0 + 0.68*w, brain_y, bw, brain_h],
    recon["R"],
    "R",
)


fig.text(
    x0 + 0.20*w,
    y0 + 0.87*h,
    "Empirical",
    ha="center",
    fontsize=10,
    fontweight="bold",
)

fig.text(
    x0 + 0.68*w,
    y0 + 0.87*h,
    "K20 reconstruction",
    ha="center",
    fontsize=10,
    fontweight="bold",
)


# Arrow
fig.text(
    x0 + 0.435*w,
    y0 + 0.61*h,
    "→",
    ha="center",
    va="center",
    fontsize=20,
)


# R2 beneath reconstruction
fig.text(
    x0 + 0.68*w,
    y0 + 0.35*h,
    (
        rf"$R^2_L={r2L:.2f}$   "
        rf"$R^2_R={r2R:.2f}$"
    ),
    ha="center",
    fontsize=10,
)


# ------------------------------------------------------------
# Residual inset
# ------------------------------------------------------------

res_y = y0 + 0.02*h
res_h = 0.26*h
res_bw = 0.15*w

add_brain_axis(
    fig,
    [
        x0 + 0.56*w,
        res_y,
        res_bw,
        res_h,
    ],
    resid["L"],
    "L",
)

add_brain_axis(
    fig,
    [
        x0 + 0.70*w,
        res_y,
        res_bw,
        res_h,
    ],
    resid["R"],
    "R",
)

fig.text(
    x0 + 0.535*w,
    y0 + 0.15*h,
    "Residual:",
    ha="right",
    va="center",
    fontsize=9,
    fontweight="bold",
)

fig.text(
    x0 + 0.69*w,
    y0 + 0.01*h,
    "empirical − K20",
    ha="center",
    fontsize=8.5,
)


# ============================================================
# Shared colorbar for B
# ============================================================

cax = fig.add_axes([
    x0 + 0.05*w,
    y0 + 0.07*h,
    0.34*w,
    0.025*h,
])

sm = mpl.cm.ScalarMappable(
    norm=norm,
    cmap=CMAP,
)

sm.set_array([])

cb = fig.colorbar(
    sm,
    cax=cax,
    orientation="horizontal",
)

cb.set_label(
    r"Spatially demeaned $\beta^*$",
    fontsize=8.5,
)

cb.ax.tick_params(
    labelsize=7,
    length=2,
)


# ============================================================
# PANEL C
#
# Placeholder axis until corrected mode-profile source is
# explicitly identified.
# ============================================================

axC = fig.add_subplot(
    gs[1, 0]
)

axC.axis("off")

axC.set_title(
    "C   Sentence-onset organization across cortical eigenmodes",
    loc="left",
    fontweight="bold",
    fontsize=12,
)

axC.text(
    0.5,
    0.52,
    (
        "Corrected mode-wise sentence-onset profile\n"
        "to be inserted from the definitive Figure-3 outputs"
    ),
    ha="center",
    va="center",
    transform=axC.transAxes,
    fontsize=11,
)

axC.text(
    0.5,
    0.34,
    "long wavelength   →   shorter wavelength",
    ha="center",
    transform=axC.transAxes,
    fontsize=9,
)


# ============================================================
# PANEL D
#
# Placeholder until we explicitly identify/load the definitive
# participant-level concentration table.
# ============================================================

axD = fig.add_subplot(
    gs[1, 1]
)

axD.axis("off")

axD.set_title(
    "D   Sentence onset is preferentially concentrated\n"
    "     in long-wavelength modes",
    loc="left",
    fontweight="bold",
    fontsize=12,
)

axD.text(
    0.5,
    0.57,
    "Corrected cumulative concentration curves",
    ha="center",
    transform=axD.transAxes,
    fontsize=11,
)

axD.text(
    0.5,
    0.39,
    (
        r"$\Delta C_{K20}=+6.14$ percentage points"
        "\n"
        r"95% bootstrap CI [4.02, 8.25]"
        "\n"
        r"$p_{\rm perm}<.0001$"
    ),
    ha="center",
    va="center",
    transform=axD.transAxes,
    fontsize=11,
)


# ============================================================
# Save draft
# ============================================================

draft_png = OUT / "sentence_onset_integrated_AB_draft.png"
draft_pdf = OUT / "sentence_onset_integrated_AB_draft.pdf"

fig.savefig(
    draft_png,
    dpi=350,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    draft_pdf,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print("\nWrote:")
print(draft_png)
print(draft_pdf)
print("\nIntegrated A/B draft complete.")