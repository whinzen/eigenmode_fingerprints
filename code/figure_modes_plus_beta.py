#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from nilearn import plotting
from nilearn.datasets import fetch_surf_fsaverage

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"
EIG_DIR = BASE / "modes" / "fsaverage5"

L_FILE = EIG_DIR / "L_phi.npy"
R_FILE = EIG_DIR / "R_phi.npy"

MODES_TO_PLOT = [1, 2, 3, 4, 5, 6]
KMAX_RECON = 20

OUT_DIR = PANG / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGRESSORS = {
    "Boundary": {
        "single": PANG / "group_sentence_level_glm" / "group_boundary_by_mode_subject_level.csv",
    },
    "Sentence shift": {
        "single": PANG / "group_sentence_level_glm" / "group_sentence_shift_by_mode_subject_level.csv",
    },
    "Token shift": {
        "L": PANG / "group_shift_glm" / "group_shift_hemi-L_by_mode_subject_level.csv",
        "R": PANG / "group_shift_glm" / "group_shift_hemi-R_by_mode_subject_level.csv",
    },
    "Prediction error": {
        "L": PANG / "group_pred_error_ar_glm" / "group_pred_error_ar_hemi-L_by_mode_subject_level.csv",
        "R": PANG / "group_pred_error_ar_glm" / "group_pred_error_ar_hemi-R_by_mode_subject_level.csv",
    },
    "Subspace exit": {
        "L": PANG / "group_pred_error_subspace_glm" / "group_pred_error_subspace_hemi-L_by_mode_subject_level.csv",
        "R": PANG / "group_pred_error_subspace_glm" / "group_pred_error_subspace_hemi-R_by_mode_subject_level.csv",
    },
    "Curvature": {
        "L": PANG / "group_curvature_glm" / "group_curvature_hemi-L_by_mode_subject_level.csv",
        "R": PANG / "group_curvature_glm" / "group_curvature_hemi-R_by_mode_subject_level.csv",
    },
}


def load_modes():
    return np.load(L_FILE), np.load(R_FILE)


def normalize(x):
    x = np.asarray(x, float)
    m = np.nanmax(np.abs(x))
    return x / m if np.isfinite(m) and m > 0 else x


def load_beta_profile_from_single(csv_file: Path):
    df = pd.read_csv(csv_file).copy()

    if "k" in df.columns and "mode_k" not in df.columns:
        df = df.rename(columns={"k": "mode_k"})

    if "hemi" not in df.columns:
        raise ValueError(f"No hemi column in {csv_file}")

    beta_L = (
        df[df["hemi"].astype(str).isin(["L", "hemi-L"])]
        .sort_values("mode_k")["beta_mean"]
        .values
    )
    beta_R = (
        df[df["hemi"].astype(str).isin(["R", "hemi-R"])]
        .sort_values("mode_k")["beta_mean"]
        .values
    )

    if len(beta_L) == 0 or len(beta_R) == 0:
        raise RuntimeError(f"Could not find both hemispheres in {csv_file}")

    return beta_L, beta_R


def load_beta_profile_from_pair(csv_L: Path, csv_R: Path):
    L = pd.read_csv(csv_L).copy()
    R = pd.read_csv(csv_R).copy()

    if "k" in L.columns and "mode_k" not in L.columns:
        L = L.rename(columns={"k": "mode_k"})
    if "k" in R.columns and "mode_k" not in R.columns:
        R = R.rename(columns={"k": "mode_k"})

    beta_L = L.sort_values("mode_k")["beta_mean"].values
    beta_R = R.sort_values("mode_k")["beta_mean"].values
    return beta_L, beta_R


def load_beta_pair(spec: dict):
    if "single" in spec:
        return load_beta_profile_from_single(spec["single"])
    return load_beta_profile_from_pair(spec["L"], spec["R"])


def reconstruct(phi, beta, kmax=20):
    """
    Reconstruct from mode-space beta coefficients.
    Excludes constant mode 0 and uses modes 1..kmax.
    """
    kmax = min(kmax, phi.shape[1] - 1, len(beta) - 1)
    return phi[:, 1:kmax + 1] @ beta[1:kmax + 1]


def plot_surface(ax, fsavg, hemi, view, data, vmax=1.0):
    if hemi == "left":
        surf = fsavg.pial_left
        bg = fsavg.sulc_left
    else:
        surf = fsavg.pial_right
        bg = fsavg.sulc_right

    plotting.plot_surf_stat_map(
        surf_mesh=surf,
        stat_map=data,
        bg_map=bg,
        hemi=hemi,
        view=view,
        axes=ax,
        cmap="RdBu_r",
        symmetric_cbar=True,
        vmax=vmax,
        threshold=None,
        colorbar=False,
        bg_on_data=True,
        darkness=None,
        title="",
    )


def add_block_headers(fig, x0, x1, y, fontsize=9):
    xs = np.linspace(x0, x1, 4, endpoint=False) + (x1 - x0) / 8
    labels = ["L lateral", "L medial", "R lateral", "R medial"]
    for x, lab in zip(xs, labels):
        fig.text(x, y, lab, ha="center", va="bottom", fontsize=fontsize)


def main():
    fsavg = fetch_surf_fsaverage(mesh="fsaverage5")
    L, R = load_modes()

    n_modes = len(MODES_TO_PLOT)
    n_regs = len(REGRESSORS)
    nrows = max(n_modes, n_regs)

    fig = plt.figure(figsize=(14, 2.0 * nrows), facecolor="white")
    gs = GridSpec(
        nrows=nrows,
        ncols=2,
        width_ratios=[1, 1],
        left=0.03,
        right=0.97,
        top=0.94,
        bottom=0.05,
        wspace=0.12,
        hspace=0.38,
    )

    fig.text(0.25, 0.975, "Low-order eigenmodes", ha="center", va="bottom", fontsize=12, fontweight="bold")
    fig.text(0.75, 0.975, "Low-pass reconstructed regressor maps", ha="center", va="bottom", fontsize=12, fontweight="bold")

    add_block_headers(fig, 0.05, 0.45, 0.952, fontsize=9)
    add_block_headers(fig, 0.55, 0.95, 0.952, fontsize=9)

    # LEFT: eigenmodes
    for i, k in enumerate(MODES_TO_PLOT):
        mode_L = normalize(L[:, k])
        mode_R = normalize(R[:, k])

        subgs = gs[i, 0].subgridspec(1, 4, wspace=0.0)

        views = [
            ("left", "lateral", mode_L),
            ("left", "medial", mode_L),
            ("right", "lateral", mode_R),
            ("right", "medial", mode_R),
        ]

        row_axes = []
        for hemi, view, data in views:
            ax = fig.add_subplot(subgs[0, len(row_axes)], projection="3d")
            plot_surface(ax, fsavg, hemi, view, data, vmax=1.0)
            row_axes.append(ax)

        y_bottom = min(ax.get_position().y0 for ax in row_axes)
        x_center = (row_axes[0].get_position().x0 + row_axes[-1].get_position().x1) / 2
        fig.text(x_center, y_bottom - 0.012, f"Mode {k}", ha="center", va="top", fontsize=10)

    # RIGHT: reconstructed maps
    for i, (name, spec) in enumerate(REGRESSORS.items()):
        beta_L, beta_R = load_beta_pair(spec)

        recon_L = normalize(reconstruct(L, beta_L, KMAX_RECON))
        recon_R = normalize(reconstruct(R, beta_R, KMAX_RECON))

        subgs = gs[i, 1].subgridspec(1, 4, wspace=0.0)

        views = [
            ("left", "lateral", recon_L),
            ("left", "medial", recon_L),
            ("right", "lateral", recon_R),
            ("right", "medial", recon_R),
        ]

        row_axes = []
        for hemi, view, data in views:
            ax = fig.add_subplot(subgs[0, len(row_axes)], projection="3d")
            plot_surface(ax, fsavg, hemi, view, data, vmax=1.0)
            row_axes.append(ax)

        y_bottom = min(ax.get_position().y0 for ax in row_axes)
        x_center = (row_axes[0].get_position().x0 + row_axes[-1].get_position().x1) / 2
        fig.text(x_center, y_bottom - 0.012, name, ha="center", va="top", fontsize=10)

    out_png = OUT_DIR / "figure_modes_plus_beta_pial.png"
    out_pdf = OUT_DIR / "figure_modes_plus_beta_pial.pdf"

    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(out_pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"✅ wrote {out_png}")
    print(f"✅ wrote {out_pdf}")


if __name__ == "__main__":
    main()