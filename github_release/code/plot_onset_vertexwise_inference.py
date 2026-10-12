
#!/usr/bin/env python
"""
Publication-quality cortical surface maps for the definitive
sentence-onset vertex-space analysis.

Panel A
-------
Observed group-mean standardized coefficient (beta*) for sentence onset.

Panel B
-------
Vertices surviving two-sided whole-cortex max-statistic permutation
FWER correction (p_FWER < .05).

Important:
- Panel A is descriptive and unthresholded.
- Panel B contains the inferential result.
- Never-valid fsaverage5 vertices are masked.
- FWER family spans both hemispheres jointly.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import nibabel as nib

from nilearn import datasets, plotting


# ============================================================
# Paths
# ============================================================

BASE = Path.home() / "eigenmode_fingerprints"

ROOT = (
    BASE
    / "pang_out"
    / "nuisance_robustness"
    / "vertex_space_linguistic_encoding"
)

QC = ROOT / "structured_null_qc"

INF = (
    ROOT
    / "onset_vertexwise_null_10000"
    / "inference"
)

OUT = INF / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load definitive empirical maps
# ============================================================

emp = {
    h: np.load(
        QC / f"sentence_onset_empirical_group_beta_star_{h}.npy"
    ).astype(float)
    for h in ["L", "R"]
}

valid = {
    h: np.load(
        ROOT / f"valid_vertex_mask_{h}.npy"
    ).astype(bool)
    for h in ["L", "R"]
}

pfwer = {
    h: np.load(
        INF / f"sentence_onset_p_FWER_maxstat_{h}.npy"
    ).astype(float)
    for h in ["L", "R"]
}


# ============================================================
# QC
# ============================================================

for h in ["L", "R"]:

    assert emp[h].shape == (10242,)
    assert valid[h].shape == (10242,)
    assert pfwer[h].shape == (10242,)

    # Never-valid vertices should not be interpreted.
    emp[h][~valid[h]] = np.nan
    pfwer[h][~valid[h]] = np.nan


# Use strict p < .05 to match definitive aggregation.
sig = {
    h: valid[h] & (pfwer[h] < 0.05)
    for h in ["L", "R"]
}

print("\nValid cortical vertices")
print("L:", valid["L"].sum())
print("R:", valid["R"].sum())

print("\nFWER-significant vertices")
print("L:", sig["L"].sum())
print("R:", sig["R"].sum())
print("Total:", sig["L"].sum() + sig["R"].sum())

print("\nMinimum FWER p")
print("L:", np.nanmin(pfwer["L"]))
print("R:", np.nanmin(pfwer["R"]))


# ============================================================
# fsaverage5 surfaces
# ============================================================

fs = datasets.fetch_surf_fsaverage(mesh="fsaverage5")


# PIAL rather than inflated:
# gives a much more conventional anatomical brain appearance.
surf = {
    "L": fs.pial_left,
    "R": fs.pial_right,
}

# Sulcal depth map provides anatomical shading.
sulc = {
    "L": fs.sulc_left,
    "R": fs.sulc_right,
}


# ============================================================
# Display parameters
# ============================================================

# Robust symmetric scale.
#
# Using the absolute 99th percentile prevents one or two extreme
# vertices from making the rest of the cortex look washed out.
all_valid = np.concatenate(
    [
        emp["L"][valid["L"]],
        emp["R"][valid["R"]],
    ]
)

vmax = np.nanpercentile(np.abs(all_valid), 99)
vmin = -vmax

print("\nDisplay scale")
print("vmin =", vmin)
print("vmax =", vmax)
print("absolute empirical maximum =",
      np.nanmax(np.abs(all_valid)))


# More saturated diverging map than the previous pale rendering.
cmap_emp = mpl.colormaps["RdBu_r"]


# ============================================================
# Figure layout
# ============================================================

# Conventional ordering:
#
#   Left lateral | Left medial | Right medial | Right lateral
#
views = [
    ("L", "lateral", "Left lateral"),
    ("L", "medial",  "Left medial"),
    ("R", "medial",  "Right medial"),
    ("R", "lateral", "Right lateral"),
]

fig = plt.figure(figsize=(16, 8.3))

# Leave space at bottom for a single horizontal colorbar.
gs = fig.add_gridspec(
    2,
    4,
    left=0.055,
    right=0.985,
    bottom=0.16,
    top=0.90,
    wspace=-0.02,
    hspace=0.06,
)


# ============================================================
# PANEL A — observed signed beta*
# ============================================================

for col, (h, view, title) in enumerate(views):

    ax = fig.add_subplot(
        gs[0, col],
        projection="3d"
    )

    plotting.plot_surf_stat_map(
        surf[h],
        emp[h],
        hemi="left" if h == "L" else "right",
        view=view,

        # Anatomical structure
        bg_map=sulc[h],
        bg_on_data=True,
        darkness=0.55,

        # Effect map
        cmap=cmap_emp,
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=True,
        threshold=None,

        # We make our own common colorbar.
        colorbar=False,

        axes=ax,
    )

    ax.set_title(
        title,
        fontsize=12,
        pad=0,
        fontweight="normal",
    )


# ============================================================
# PANEL B — FWER-significant vertices
# ============================================================

for col, (h, view, title) in enumerate(views):

    ax = fig.add_subplot(
        gs[1, col],
        projection="3d"
    )

    # First draw anatomy alone.
    #
    # plot_surf accepts a constant texture here; with alpha=0
    # we retain the anatomical sulcal background.
    base = np.zeros(10242)

    plotting.plot_surf_stat_map(
        surf[h],
        base,
        hemi="left" if h == "L" else "right",
        view=view,

        bg_map=sulc[h],
        bg_on_data=False,
        darkness=0.55,

        cmap="Greys",
        threshold=1,       # hides the zero statistical map
        colorbar=False,

        axes=ax,
    )

    # Overlay significant vertices as a discrete map.
    #
    # NaN = not significant.
    # 1   = survives whole-cortex FWER.
    overlay = np.full(10242, np.nan)
    overlay[sig[h]] = 1.0

    if np.any(sig[h]):

        plotting.plot_surf_stat_map(
            surf[h],
            overlay,
            hemi="left" if h == "L" else "right",
            view=view,

            # No second anatomical background:
            bg_map=None,

            cmap="autumn",
            vmin=0,
            vmax=1,
            threshold=0.5,
            colorbar=False,

            axes=ax,
        )


# ============================================================
# Panel labels / row labels
# ============================================================

fig.text(
    0.018,
    0.695,
    "A",
    fontsize=18,
    fontweight="bold",
    va="center",
)

fig.text(
    0.040,
    0.695,
    "Observed sentence-onset response",
    fontsize=13,
    rotation=90,
    va="center",
)

fig.text(
    0.018,
    0.335,
    "B",
    fontsize=18,
    fontweight="bold",
    va="center",
)

fig.text(
    0.040,
    0.335,
    "Whole-cortex FWER inference",
    fontsize=13,
    rotation=90,
    va="center",
)


# ============================================================
# Shared beta* colorbar
# ============================================================

norm = mpl.colors.Normalize(
    vmin=vmin,
    vmax=vmax,
)

sm = mpl.cm.ScalarMappable(
    norm=norm,
    cmap=cmap_emp,
)
sm.set_array([])

cax = fig.add_axes(
    [0.35, 0.095, 0.30, 0.022]
)

cb = fig.colorbar(
    sm,
    cax=cax,
    orientation="horizontal",
)

cb.set_label(
    r"Sentence-onset standardized coefficient ($\beta^*$)",
    fontsize=11,
)

cb.ax.tick_params(
    labelsize=9,
    length=3,
)


# ============================================================
# Inferential annotation
# ============================================================

nL = int(sig["L"].sum())
nR = int(sig["R"].sum())

fig.text(
    0.5,
    0.035,
    (
        "Panel B: vertices surviving two-sided max-statistic "
        "permutation correction across both hemispheres "
        f"(FWER < .05; left: {nL}, right: {nR})."
    ),
    ha="center",
    va="center",
    fontsize=10,
)


# ============================================================
# Save
# ============================================================

png = OUT / "sentence_onset_vertexwise_FWER_pial.png"
pdf = OUT / "sentence_onset_vertexwise_FWER_pial.pdf"

fig.savefig(
    png,
    dpi=400,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    pdf,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print("\nWrote:")
print(png)
print(pdf)