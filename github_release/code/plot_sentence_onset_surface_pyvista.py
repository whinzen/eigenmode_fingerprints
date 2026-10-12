#!/usr/bin/env python
"""
Publication-quality sentence-onset cortical surface map.

Produces:
1. Smooth PyVista/VTK rendering of the unthresholded group beta* map.
2. FWER-significant vertices marked directly on the same surface.
3. Connected-component analysis of significant vertices on the
   fsaverage5 mesh.

Inference:
- empirical map = group-mean standardized coefficient beta*
- significance = two-sided max-statistic permutation FWER p < .05
- FWER family spans both hemispheres jointly
- never-valid vertices are excluded

This is intended as the cortical-map component of the final figure,
not as a standalone inferential figure.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import colormaps
from nilearn import datasets
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


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

OUT = INF / "figures_pyvista"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# PyVista configuration
# ============================================================

pv.global_theme.allow_empty_mesh = True

# Important on a cluster / headless node.
pv.OFF_SCREEN = True


# ============================================================
# Load empirical maps and inference
# ============================================================

emp = {}
valid = {}
pfwer = {}
sig = {}

for h in ["L", "R"]:

    emp[h] = np.load(
        QC / f"sentence_onset_empirical_group_beta_star_{h}.npy"
    ).astype(float)

    valid[h] = np.load(
        ROOT / f"valid_vertex_mask_{h}.npy"
    ).astype(bool)

    pfwer[h] = np.load(
        INF / f"sentence_onset_p_FWER_maxstat_{h}.npy"
    ).astype(float)

    assert emp[h].shape == (10242,)
    assert valid[h].shape == (10242,)
    assert pfwer[h].shape == (10242,)

    sig[h] = (
        valid[h]
        & np.isfinite(pfwer[h])
        & (pfwer[h] < 0.05)
    )


print("\nFWER-significant vertices")
print("L:", sig["L"].sum())
print("R:", sig["R"].sum())
print("Total:", sig["L"].sum() + sig["R"].sum())


# ============================================================
# Fetch fsaverage5 geometry
# ============================================================

fs = datasets.fetch_surf_fsaverage(mesh="fsaverage5")


# ------------------------------------------------------------
# Surface choice
#
# Use INFLATED geometry for visibility, but with:
#   - smooth VTK shading
#   - curvature/sulcal information encoded subtly in geometry
#   - no mesh edges
#
# This should look much more like a standard neuroimaging surface
# than the previous Matplotlib/Nilearn rendering.
# ------------------------------------------------------------

surface_files = {
    "L": fs.infl_left,
    "R": fs.infl_right,
}

sulc_files = {
    "L": fs.sulc_left,
    "R": fs.sulc_right,
}


# ============================================================
# Helpers
# ============================================================

def load_surface(fname):
    """
    Read GIFTI surface and convert to PyVista PolyData.
    """
    g = nib.load(fname)

    coords = np.asarray(g.darrays[0].data, dtype=float)
    faces = np.asarray(g.darrays[1].data, dtype=np.int64)

    # PyVista faces:
    # [3, i, j, k, 3, i, j, k, ...]
    pv_faces = np.column_stack(
        [
            np.full(len(faces), 3, dtype=np.int64),
            faces,
        ]
    ).ravel()

    mesh = pv.PolyData(coords, pv_faces)

    return mesh, coords, faces


def load_texture(fname):
    g = nib.load(fname)
    return np.asarray(g.darrays[0].data, dtype=float)


# ============================================================
# Load meshes
# ============================================================

mesh = {}
coords = {}
faces = {}
sulc = {}

for h in ["L", "R"]:

    mesh[h], coords[h], faces[h] = load_surface(
        surface_files[h]
    )

    sulc[h] = load_texture(
        sulc_files[h]
    )

    assert mesh[h].n_points == 10242


# ============================================================
# Connected-component analysis of significant vertices
# ============================================================

def significant_components(faces, sigmask):
    """
    Connected components among significant vertices using
    triangular-mesh adjacency.
    """

    nvert = sigmask.size

    # Mesh edges
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]

    edges = np.vstack([e01, e12, e20])

    # Undirected adjacency
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])

    data = np.ones(len(rows), dtype=np.uint8)

    adj = coo_matrix(
        (data, (rows, cols)),
        shape=(nvert, nvert),
    ).tocsr()

    idx = np.where(sigmask)[0]

    if len(idx) == 0:
        return []

    sub = adj[idx][:, idx]

    ncomp, labels = connected_components(
        sub,
        directed=False,
        return_labels=True,
    )

    comps = []

    for c in range(ncomp):
        verts = idx[labels == c]
        comps.append(verts)

    comps.sort(
        key=len,
        reverse=True,
    )

    return comps


component_rows = []

for h in ["L", "R"]:

    comps = significant_components(
        faces[h],
        sig[h],
    )

    print(f"\n{h} hemisphere:")
    print("Number of FWER components:", len(comps))

    for ci, verts in enumerate(comps, start=1):

        peak_local = verts[
            np.argmax(np.abs(emp[h][verts]))
        ]

        print(
            f"  Component {ci}: "
            f"{len(verts)} vertices; "
            f"peak vertex={peak_local}; "
            f"beta*={emp[h][peak_local]:.6f}; "
            f"p_FWER={pfwer[h][peak_local]:.6f}"
        )

        component_rows.append(
            {
                "hemisphere": h,
                "component": ci,
                "n_vertices": len(verts),
                "peak_vertex": int(peak_local),
                "peak_beta_star": emp[h][peak_local],
                "peak_p_FWER": pfwer[h][peak_local],
            }
        )


pd.DataFrame(component_rows).to_csv(
    OUT / "sentence_onset_FWER_components.csv",
    index=False,
)


# ============================================================
# Display scale
# ============================================================

all_beta = np.concatenate(
    [
        emp["L"][valid["L"]],
        emp["R"][valid["R"]],
    ]
)

# Robust scale so widespread cortical effects remain visible.
vmax = np.nanpercentile(
    np.abs(all_beta),
    99
)

vmin = -vmax

print("\nDisplay limits:")
print(vmin, vmax)

norm = TwoSlopeNorm(
    vmin=vmin,
    vcenter=0,
    vmax=vmax,
)

cmap = colormaps["RdBu_r"]


# ============================================================
# Camera helpers
# ============================================================

def set_camera(plotter, hemi, view):
    """
    Standard FreeSurfer-like lateral / medial orientation.

    We use explicit cameras so all panels are reproducible.
    """

    plotter.camera.focal_point = (0, 0, 0)

    if hemi == "L" and view == "lateral":
        plotter.camera.position = (-400, 0, 0)
        plotter.camera.up = (0, 0, 1)

    elif hemi == "L" and view == "medial":
        plotter.camera.position = (400, 0, 0)
        plotter.camera.up = (0, 0, 1)

    elif hemi == "R" and view == "medial":
        plotter.camera.position = (-400, 0, 0)
        plotter.camera.up = (0, 0, 1)

    elif hemi == "R" and view == "lateral":
        plotter.camera.position = (400, 0, 0)
        plotter.camera.up = (0, 0, 1)

    plotter.camera.zoom(1.55)


# ============================================================
# Rendering
# ============================================================

views = [
    ("L", "lateral", "Left lateral"),
    ("L", "medial",  "Left medial"),
    ("R", "medial",  "Right medial"),
    ("R", "lateral", "Right lateral"),
]


plotter = pv.Plotter(
    shape=(1, 4),
    off_screen=True,
    window_size=(2400, 650),
    border=False,
)


for col, (h, view, title) in enumerate(views):

    plotter.subplot(0, col)

    m = mesh[h].copy()

    # --------------------------------------------------------
    # Convert beta* values to RGB ourselves.
    #
    # This allows invalid cortex to be neutral grey while
    # retaining a saturated diverging map elsewhere.
    # --------------------------------------------------------

    beta = emp[h].copy()

    clipped = np.clip(
        beta,
        vmin,
        vmax,
    )

    rgba = cmap(
        norm(clipped)
    )

    rgb = (
        rgba[:, :3] * 255
    ).astype(np.uint8)

    # Never-valid vertices = neutral grey.
    rgb[~valid[h]] = np.array(
        [190, 190, 190],
        dtype=np.uint8,
    )

    m.point_data["RGB"] = rgb

    # --------------------------------------------------------
    # Main cortical surface
    # --------------------------------------------------------

    plotter.add_mesh(
        m,
        scalars="RGB",
        rgb=True,

        # Critical visual settings:
        smooth_shading=True,
        show_edges=False,

        ambient=0.35,
        diffuse=0.75,
        specular=0.08,
        specular_power=10,
    )

    # --------------------------------------------------------
    # FWER-significant vertices
    #
    # Small black spheres provide an inferential marker without
    # replacing the underlying beta* colour.
    # --------------------------------------------------------

    idx = np.where(sig[h])[0]

    if len(idx):

        pts = coords[h][idx]

        pdata = pv.PolyData(pts)

        spheres = pdata.glyph(
            geom=pv.Sphere(
                radius=1.6,
                theta_resolution=20,
                phi_resolution=20,
            ),
            scale=False,
            orient=False,
        )

        plotter.add_mesh(
            spheres,
            color="black",
            smooth_shading=True,
            ambient=0.5,
            diffuse=0.5,
        )

    plotter.add_text(
        title,
        position="upper_edge",
        font_size=13,
        color="black",
    )

    plotter.set_background("white")

    set_camera(
        plotter,
        h,
        view,
    )


# ============================================================
# Screenshot
# ============================================================

png = OUT / "sentence_onset_beta_star_with_FWER.png"

plotter.screenshot(
    str(png),
    transparent_background=False,
)

plotter.close()

print("\nSaved surface rendering:")
print(png)


# ============================================================
# Add publication colorbar with Matplotlib
#
# We do this separately because Matplotlib produces a cleaner
# publication-quality vector colorbar than the VTK scalar bar.
# ============================================================

img = plt.imread(png)

fig = plt.figure(
    figsize=(16, 4.9)
)

ax = fig.add_axes(
    [0.01, 0.16, 0.98, 0.81]
)

ax.imshow(img)
ax.axis("off")


# Shared horizontal colour bar
cax = fig.add_axes(
    [0.36, 0.075, 0.28, 0.035]
)

sm = plt.cm.ScalarMappable(
    norm=norm,
    cmap=cmap,
)
sm.set_array([])

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
)


# Small inferential key
fig.text(
    0.5,
    0.015,
    (
        "Black markers: vertices surviving two-sided "
        "whole-cortex max-statistic permutation correction "
        "(FWER < .05)"
    ),
    ha="center",
    fontsize=9,
)


final_png = (
    OUT
    / "sentence_onset_beta_star_with_FWER_FINAL.png"
)

final_pdf = (
    OUT
    / "sentence_onset_beta_star_with_FWER_FINAL.pdf"
)

fig.savefig(
    final_png,
    dpi=400,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    final_pdf,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


print("\nFinal files:")
print(final_png)
print(final_pdf)

print("\nDONE")