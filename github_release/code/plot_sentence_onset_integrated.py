#!/usr/bin/env python
"""
Integrated definitive sentence-onset cortical/eigenmode figure.

A  Conventional vertex-space validation against the structured
   stimulus-preserving temporal null.

B  Corrected empirical cortical sentence-onset map ->
   K20 eigenmode reconstruction -> residual.
   Empirical, reconstruction and residual use the SAME colour scale.

C  Cumulative concentration of sentence-onset |beta| compared with
   the corrected empirical cortical modal-energy spectrum.

D  Participant-paired excess concentration:
       D(K) = C_beta(K) - C_energy(K)
   with 95% bootstrap confidence intervals from the frozen corrected
   inference outputs.

All numerical results are loaded from definitive nuisance-corrected
outputs. No inferential statistics are recomputed here.
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
    BASE
    / "pang_out"
    / "nuisance_robustness"
    / "vertex_space_linguistic_encoding"
)

RECON = (
    VROOT
    / "sentence_onset_K20_reconstruction"
)

NULLDIR = (
    VROOT
    / "onset_vertexwise_null_10000"
)

STRUCT_SUMMARY = (
    VROOT
    / "structured_null_10000"
    / "vertex_structured_null_10000_summary.csv"
)

CORR = (
    BASE
    / "pang_out"
    / "nuisance_robustness"
    / "corrected_validated_linguistic_effects"
)

CONC_PART = (
    CORR
    / "corrected_participant_concentration.csv"
)

CONC_INF = (
    CORR
    / "corrected_concentration_inference_all_K.csv"
)

OUT = (
    BASE
    / "pang_out"
    / "paper_figures"
    / "sentence_onset_integrated"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load corrected empirical / K20 / residual maps
# ============================================================

emp = {
    h: np.load(
        RECON
        / f"sentence_onset_empirical_demeaned_{h}.npy"
    ).astype(float)
    for h in ["L", "R"]
}

recon = {
    h: np.load(
        RECON
        / f"sentence_onset_K20_reconstruction_{h}.npy"
    ).astype(float)
    for h in ["L", "R"]
}

resid = {
    h: np.load(
        RECON
        / f"sentence_onset_K20_residual_{h}.npy"
    ).astype(float)
    for h in ["L", "R"]
}

valid = {
    h: np.load(
        VROOT
        / f"valid_vertex_mask_{h}.npy"
    ).astype(bool)
    for h in ["L", "R"]
}


# ============================================================
# Reconstruction summary
# ============================================================

recon_summary = pd.read_csv(
    RECON
    / "sentence_onset_K20_reconstruction_summary.csv"
)

r2L = float(
    recon_summary.loc[
        recon_summary["hemisphere"] == "L",
        "R2_K20"
    ].iloc[0]
)

r2R = float(
    recon_summary.loc[
        recon_summary["hemisphere"] == "R",
        "R2_K20"
    ].iloc[0]
)


print("\nK20 reconstruction")
print("R2 L:", r2L)
print("R2 R:", r2R)


# ============================================================
# Reconstruction identity QC
# ============================================================

for h in ["L", "R"]:

    assert emp[h].shape == (10242,)
    assert recon[h].shape == (10242,)
    assert resid[h].shape == (10242,)
    assert valid[h].shape == (10242,)

    d = (
        emp[h][valid[h]]
        - recon[h][valid[h]]
        - resid[h][valid[h]]
    )

    maxerr = np.max(
        np.abs(d)
    )

    print(
        f"{h} reconstruction identity "
        f"max abs error: {maxerr:.3e}"
    )

    assert maxerr < 1e-10


# ============================================================
# PANEL A
# Definitive structured-null validation
# ============================================================

rms_files = sorted(
    NULLDIR.glob(
        "rms_*.csv"
    )
)

assert len(rms_files) == 100

rms_null = pd.concat(
    [
        pd.read_csv(f)
        for f in rms_files
    ],
    ignore_index=True,
).sort_values("perm")

assert len(rms_null) == 10000
assert rms_null["perm"].nunique() == 10000

null_rms = (
    rms_null[
        "group_map_rms"
    ]
    .to_numpy()
    .astype(float)
)


# Frozen inferential summary
ss = pd.read_csv(
    STRUCT_SUMMARY
)

onset_row = ss.loc[
    ss["predictor"]
    == "sentence_onset"
].iloc[0]

EMP_RMS_DISPLAY = float(
    onset_row[
        "empirical_group_map_rms"
    ]
)

NULL_MEAN_FROZEN = float(
    onset_row[
        "null_mean"
    ]
)

N_GE_FROZEN = int(
    onset_row[
        "n_null_ge_empirical"
    ]
)

P_RMS = float(
    onset_row[
        "p_perm_greater"
    ]
)


assert N_GE_FROZEN == 359

assert np.isclose(
    P_RMS,
    0.035996,
    atol=1e-6,
)

assert np.isclose(
    null_rms.mean(),
    NULL_MEAN_FROZEN,
    atol=1e-6,
)


print("\nVertex-space validation")
print(
    "Empirical RMS:",
    EMP_RMS_DISPLAY,
)
print(
    "Frozen p:",
    P_RMS,
)


# ============================================================
# PANELS C/D
# Corrected participant concentration data
# ============================================================

conc = pd.read_csv(
    CONC_PART
)

inf = pd.read_csv(
    CONC_INF
)


# Sentence onset only
conc = conc.loc[
    conc["predictor"]
    == "Sentence onset"
].copy()

inf = inf.loc[
    inf["predictor"]
    == "Sentence onset"
].copy()


# Required K values
KS = np.array(
    [5, 10, 20, 30, 50]
)

assert set(
    conc["K"].unique()
) == set(KS)

assert set(
    inf["K"].unique()
) == set(KS)

assert conc["subject"].nunique() == 49


# ------------------------------------------------------------
# Participant means for C
# ------------------------------------------------------------

curve = (
    conc.groupby("K")[
        ["C_beta", "C_energy"]
    ]
    .mean()
    .reindex(KS)
)

mean_Cbeta = (
    curve["C_beta"]
    .to_numpy()
)

mean_Cenergy = (
    curve["C_energy"]
    .to_numpy()
)


# ------------------------------------------------------------
# Frozen inference for D
# ------------------------------------------------------------

inf = (
    inf
    .set_index("K")
    .reindex(KS)
)

mean_delta = (
    inf["mean_delta"]
    .to_numpy()
)

ci_lo = (
    inf["ci_lo"]
    .to_numpy()
)

ci_hi = (
    inf["ci_hi"]
    .to_numpy()
)

p_perm_K = (
    inf["p_perm"]
    .to_numpy()
)


# Cross-check participant means against inference table
assert np.allclose(
    mean_Cbeta,
    inf["mean_C_beta"].to_numpy(),
)

assert np.allclose(
    mean_Cenergy,
    inf["mean_C_energy"].to_numpy(),
)

assert np.allclose(
    mean_Cbeta - mean_Cenergy,
    mean_delta,
)


# K20 values
i20 = np.where(
    KS == 20
)[0][0]

Cbeta20 = mean_Cbeta[i20]
Cenergy20 = mean_Cenergy[i20]
delta20 = mean_delta[i20]
lo20 = ci_lo[i20]
hi20 = ci_hi[i20]
p20 = p_perm_K[i20]


print("\nCorrected concentration at K20")
print(
    "C_beta:",
    Cbeta20,
)
print(
    "C_energy:",
    Cenergy20,
)
print(
    "Delta:",
    delta20,
)
print(
    "95% CI:",
    lo20,
    hi20,
)
print(
    "p:",
    p20,
)


# Definitive result QC
assert np.isclose(
    delta20,
    0.061398,
    atol=1e-6,
)

assert np.isclose(
    lo20,
    0.040331,
    atol=1e-6,
)

assert np.isclose(
    hi20,
    0.082512,
    atol=1e-6,
)


print(
    "Concentration provenance QC PASSED"
)


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


# ============================================================
# Common colour scale for B
# ============================================================

all_emp = np.concatenate(
    [
        emp["L"][
            valid["L"]
        ],
        emp["R"][
            valid["R"]
        ],
    ]
)

VMAX = np.nanpercentile(
    np.abs(all_emp),
    99,
)

VMIN = -VMAX

norm = mpl.colors.TwoSlopeNorm(
    vmin=VMIN,
    vcenter=0,
    vmax=VMAX,
)

CMAP = "RdBu_r"


# ============================================================
# Surface helper
# ============================================================

def add_brain_axis(
    fig,
    rect,
    data,
    hemi,
    view="lateral",
):

    ax = fig.add_axes(
        rect,
        projection="3d",
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
# Figure
# ============================================================

fig = plt.figure(
    figsize=(15.5, 9.0),
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

    width_ratios=[
        1.03,
        1.17,
    ],

    height_ratios=[
        1.0,
        1.0,
    ],

    wspace=0.25,
    hspace=0.32,
)


# ============================================================
# PANEL A
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
    alpha=0.78,
)

axA.axvline(
    EMP_RMS_DISPLAY,
    linewidth=2.3,
)

ylim = axA.get_ylim()

axA.text(
    EMP_RMS_DISPLAY,
    ylim[1] * 0.88,
    " empirical",
    rotation=90,
    va="top",
    ha="left",
    fontsize=10,
)

axA.text(
    0.97,
    0.94,
    (
        rf"$p_{{\rm perm}}"
        rf"={P_RMS:.3f}$"
        "\n"
        r"$N_{\rm perm}=10{,}000$"
    ),
    transform=axA.transAxes,
    ha="right",
    va="top",
    fontsize=10,
)

axA.set_xlabel(
    r"Whole-cortex group-map RMS of $\beta^*$"
)

axA.set_ylabel(
    "Null density"
)

axA.set_title(
    "A   Sentence-onset response exceeds "
    "the structured temporal null",
    loc="left",
    fontweight="bold",
    fontsize=12,
)

axA.spines[
    "top"
].set_visible(False)

axA.spines[
    "right"
].set_visible(False)


# ============================================================
# PANEL B
# Empirical -> K20 -> residual
# ============================================================

axB = fig.add_subplot(
    gs[0, 1]
)

axB.axis("off")

axB.set_title(
    "B   Low-order eigenmodes preserve "
    "broad spatial structure",
    loc="left",
    fontweight="bold",
    fontsize=12,
    pad=8,
)

bbox = axB.get_position()

x0 = bbox.x0
y0 = bbox.y0
w = bbox.width
h = bbox.height


# Larger main brains than previous draft
brain_y = (
    y0
    + 0.35*h
)

brain_h = (
    0.53*h
)

brain_w = (
    0.23*w
)


# Empirical
add_brain_axis(
    fig,
    [
        x0 + 0.00*w,
        brain_y,
        brain_w,
        brain_h,
    ],
    emp["L"],
    "L",
)

add_brain_axis(
    fig,
    [
        x0 + 0.21*w,
        brain_y,
        brain_w,
        brain_h,
    ],
    emp["R"],
    "R",
)


# K20
add_brain_axis(
    fig,
    [
        x0 + 0.54*w,
        brain_y,
        brain_w,
        brain_h,
    ],
    recon["L"],
    "L",
)

add_brain_axis(
    fig,
    [
        x0 + 0.75*w,
        brain_y,
        brain_w,
        brain_h,
    ],
    recon["R"],
    "R",
)


fig.text(
    x0 + 0.215*w,
    y0 + 0.89*h,
    "Empirical",
    ha="center",
    fontsize=10,
    fontweight="bold",
)

fig.text(
    x0 + 0.755*w,
    y0 + 0.89*h,
    "K20 reconstruction",
    ha="center",
    fontsize=10,
    fontweight="bold",
)


fig.text(
    x0 + 0.485*w,
    y0 + 0.62*h,
    "→",
    ha="center",
    va="center",
    fontsize=20,
)


fig.text(
    x0 + 0.755*w,
    y0 + 0.345*h,
    (
        rf"$R^2_L={r2L:.2f}$"
        "     "
        rf"$R^2_R={r2R:.2f}$"
    ),
    ha="center",
    fontsize=10,
)


# ------------------------------------------------------------
# Residual inset
# ------------------------------------------------------------

res_y = (
    y0
    + 0.025*h
)

res_h = (
    0.245*h
)

res_w = (
    0.15*w
)

add_brain_axis(
    fig,
    [
        x0 + 0.61*w,
        res_y,
        res_w,
        res_h,
    ],
    resid["L"],
    "L",
)

add_brain_axis(
    fig,
    [
        x0 + 0.755*w,
        res_y,
        res_w,
        res_h,
    ],
    resid["R"],
    "R",
)


fig.text(
    x0 + 0.58*w,
    y0 + 0.145*h,
    "Residual:",
    ha="right",
    va="center",
    fontsize=9,
    fontweight="bold",
)

fig.text(
    x0 + 0.755*w,
    y0 + 0.005*h,
    "empirical − K20",
    ha="center",
    fontsize=8.5,
)


# Shared B colorbar
cax = fig.add_axes(
    [
        x0 + 0.04*w,
        y0 + 0.065*h,
        0.37*w,
        0.025*h,
    ]
)

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
# Cumulative concentration curves
# ============================================================

axC = fig.add_subplot(
    gs[1, 0]
)


# Participant trajectories very faintly in background
for subject, df in conc.groupby(
    "subject"
):

    df = (
        df
        .set_index("K")
        .reindex(KS)
    )

    axC.plot(
        KS,
        df["C_beta"],
        linewidth=0.45,
        alpha=0.08,
    )


# Group curves
line_beta, = axC.plot(
    KS,
    mean_Cbeta,
    marker="o",
    linewidth=2.4,
    markersize=5.5,
    label=r"Sentence onset $|\beta|$",
)

line_energy, = axC.plot(
    KS,
    mean_Cenergy,
    marker="o",
    linewidth=2.4,
    markersize=5.5,
    label="Empirical cortical energy",
)


# K20 guide
axC.axvline(
    20,
    linestyle="--",
    linewidth=1.0,
    alpha=0.55,
)


# Highlight K20 gap
axC.plot(
    [20, 20],
    [
        Cenergy20,
        Cbeta20,
    ],
    linewidth=3,
)

axC.text(
    21.5,
    (
        Cbeta20
        + Cenergy20
    ) / 2,
    "+6.14 pp",
    va="center",
    fontsize=9,
)


axC.set_xlabel(
    "Number of nonconstant eigenmodes, K"
)

axC.set_ylabel(
    "Cumulative concentration"
)

axC.set_xticks(
    KS
)

axC.set_ylim(
    0.15,
    0.82,
)

axC.legend(
    frameon=False,
    fontsize=9,
    loc="lower right",
)

axC.set_title(
    "C   Sentence onset is concentrated "
    "toward low-order modes",
    loc="left",
    fontweight="bold",
    fontsize=12,
)

axC.spines[
    "top"
].set_visible(False)

axC.spines[
    "right"
].set_visible(False)


# ============================================================
# PANEL D
# Paired excess concentration
# ============================================================

axD = fig.add_subplot(
    gs[1, 1]
)


# Convert to percentage points
delta_pp = (
    mean_delta * 100
)

lo_pp = (
    ci_lo * 100
)

hi_pp = (
    ci_hi * 100
)

yerr = np.vstack(
    [
        delta_pp - lo_pp,
        hi_pp - delta_pp,
    ]
)


axD.axhline(
    0,
    linewidth=1,
    linestyle="--",
    alpha=0.55,
)


axD.errorbar(
    KS,
    delta_pp,
    yerr=yerr,

    marker="o",
    markersize=6,
    linewidth=2.1,
    capsize=4,
)


# Highlight K20
axD.scatter(
    [20],
    [delta_pp[i20]],
    s=75,
    zorder=5,
)


axD.annotate(
    (
        "+6.14 pp at K=20\n"
        "95% CI [4.02, 8.25]\n"
        r"$p_{\rm perm}<.0001$"
    ),

    xy=(
        20,
        delta_pp[i20],
    ),

    xytext=(
        28,
        8.7,
    ),

    arrowprops=dict(
        arrowstyle="-",
        linewidth=0.8,
        shrinkA=2,
        shrinkB=4,
    ),

    fontsize=9,
    va="top",
    ha="left",
    linespacing=1.35,
)

axD.set_xlabel(
    "Number of nonconstant eigenmodes, K"
)

axD.set_ylabel(
    (
        "Excess concentration over "
        "empirical cortex\n"
        r"$C_{\beta}(K)-C_{\rm energy}(K)$ "
        "(percentage points)"
    )
)

axD.set_xticks(
    KS
)

axD.set_ylim(
    -0.5,
    9.5,
)

axD.set_title(
    "D   Preferential long-wavelength "
    "concentration persists across K",
    loc="left",
    fontweight="bold",
    fontsize=12,
)

axD.spines[
    "top"
].set_visible(False)

axD.spines[
    "right"
].set_visible(False)


# ============================================================
# Save
# ============================================================

PNG = (
    OUT
    / "sentence_onset_integrated_FINAL.png"
)

PDF = (
    OUT
    / "sentence_onset_integrated_FINAL.pdf"
)


fig.savefig(
    PNG,
    dpi=400,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    PDF,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


# ============================================================
# Terminal summary
# ============================================================

print("\n========================================")
print("FINAL INTEGRATED FIGURE COMPLETE")
print("========================================")

print("\nA — vertex-space validation")
print(
    "empirical RMS:",
    EMP_RMS_DISPLAY,
)
print(
    "p_perm:",
    P_RMS,
)

print("\nB — K20 reconstruction")
print(
    "R2 L:",
    r2L,
)
print(
    "R2 R:",
    r2R,
)

print("\nC/D — low-mode concentration")
print(
    "K20 C_beta:",
    Cbeta20,
)
print(
    "K20 C_energy:",
    Cenergy20,
)
print(
    "K20 excess pp:",
    delta20 * 100,
)
print(
    "95% CI pp:",
    lo20 * 100,
    hi20 * 100,
)
print(
    "p_perm:",
    p20,
)

print("\nWrote:")
print(PNG)
print(PDF)