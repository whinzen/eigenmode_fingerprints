#!/usr/bin/env python

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from nilearn import datasets, plotting

BASE = Path.home() / "eigenmode_fingerprints"
MODE_DIR = BASE / "modes" / "fsaverage5"
BETA_DIR = BASE / "pang_out" / "vertex_betas"
OUT_FIG = BASE / "pang_out" / "paper_figures"
OUT_TAB = BASE / "pang_out" / "paper_tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

LABELS = {
    "sentence_shift": "Sentence shift",
    "boundary": "Sentence boundary",
    "token_shift": "Token shift",
    "pred_error_ar": "Prediction error",
    "pred_error_subspace": "Subspace exit",
    "curvature": "Curvature",
    "wordrate": "Word rate",
    "content_density": "Content-word density",
}


def load_phi(hemi):
    return np.load(MODE_DIR / f"{hemi}_phi.npy")


def load_beta(metric, hemi):
    f = BETA_DIR / f"{metric}_{hemi}.npy"
    if not f.exists():
        raise FileNotFoundError(f)
    x = np.load(f).astype(float)

    # Treat exact zeros as medial-wall / missing-data vertices.
    # In these maps, true beta values are continuous; exact zeros are mask-like.
    x[x == 0] = np.nan
    return x


def reconstruct(beta_map, phi, k_low):
    beta_map = np.asarray(beta_map, float)
    mask = np.isfinite(beta_map)

    if mask.sum() < 100:
        raise RuntimeError("Too few valid vertices after masking.")

    # Spatially demean valid cortex only.
    y = np.full_like(beta_map, np.nan, dtype=float)
    y[mask] = beta_map[mask] - np.nanmean(beta_map[mask])

    # Use least-squares projection because phi is not guaranteed to be
    # orthonormal under the masked vertex metric.
    B_full = phi[:, 1:]  # exclude constant mode
    coef_full, _, _, _ = np.linalg.lstsq(B_full[mask], y[mask], rcond=None)

    full = np.full_like(y, np.nan)
    full[mask] = B_full[mask] @ coef_full

    k_low = min(k_low, phi.shape[1] - 1)
    B_low = phi[:, 1:k_low + 1]
    coef_low, _, _, _ = np.linalg.lstsq(B_low[mask], y[mask], rcond=None)

    low = np.full_like(y, np.nan)
    low[mask] = B_low[mask] @ coef_low

    resid = np.full_like(y, np.nan)
    resid[mask] = y[mask] - low[mask]

    return y, full, low, resid, mask


def stats(emp, recon):
    good = np.isfinite(emp) & np.isfinite(recon)
    x = emp[good]
    y = recon[good]

    if len(x) < 10:
        return np.nan, np.nan

    r = np.corrcoef(x, y)[0, 1]
    ss_res = np.sum((x - y) ** 2)
    ss_tot = np.sum((x - x.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return r, r2


def plot_brain(ax, surf, data, bg, hemi, view, vmax, title=""):
    plotting.plot_surf_stat_map(
        surf_mesh=surf,
        stat_map=data,
        bg_map=bg,
        hemi=hemi,
        view=view,
        axes=ax,
        colorbar=False,
        cmap="RdBu_r",
        symmetric_cbar=True,
        vmax=vmax,
        threshold=None,
        bg_on_data=True,
        darkness=None,
        title=title,
    )


def add_colorbar(fig, cax, vmax, label):
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)


def robust_vmax(arrays, pct=99):
    x = np.concatenate([a[np.isfinite(a)] for a in arrays])
    return np.nanpercentile(np.abs(x), pct)


def make_figure(metric, k_low):
    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    phi_L = load_phi("L")
    phi_R = load_phi("R")

    emp_L_raw = load_beta(metric, "L")
    emp_R_raw = load_beta(metric, "R")

    emp_L, full_L, low_L, res_L, mask_L = reconstruct(emp_L_raw, phi_L, k_low)
    emp_R, full_R, low_R, res_R, mask_R = reconstruct(emp_R_raw, phi_R, k_low)

    emp = np.concatenate([emp_L, emp_R])
    full = np.concatenate([full_L, full_R])
    low = np.concatenate([low_L, low_R])
    res = np.concatenate([res_L, res_R])

    r_full, r2_full = stats(emp, full)
    r_low, r2_low = stats(emp, low)

    vmax_main = robust_vmax([emp_L, emp_R, full_L, full_R, low_L, low_R], pct=99)
    vmax_res = robust_vmax([res_L, res_R], pct=99)

    rows = [
        ("A. Empirical\nvertexwise β map\n(demeaned)", emp_L, emp_R, vmax_main),
        ("B. Full retained-mode\nreconstruction", full_L, full_R, vmax_main),
        (f"C. Low-pass reconstruction\nmodes 1–{k_low}", low_L, low_R, vmax_main),
        ("D. Residual\nempirical − low-pass", res_L, res_R, vmax_res),
    ]

    fig = plt.figure(figsize=(13, 10.2))
    gs = fig.add_gridspec(
        nrows=4,
        ncols=6,
        width_ratios=[1.35, 1, 1, 1, 1, 0.07],
        wspace=0.02,
        hspace=0.02,
    )

    for i, (row_label, data_L, data_R, vmax) in enumerate(rows):
        lab_ax = fig.add_subplot(gs[i, 0])
        lab_ax.axis("off")
        lab_ax.text(
            0.98,
            0.38,
            row_label,
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

        axs = [
            fig.add_subplot(gs[i, 1], projection="3d"),
            fig.add_subplot(gs[i, 2], projection="3d"),
            fig.add_subplot(gs[i, 3], projection="3d"),
            fig.add_subplot(gs[i, 4], projection="3d"),
        ]

        titles = ["L lateral", "L medial", "R lateral", "R medial"] if i == 0 else ["", "", "", ""]

        plot_brain(axs[0], fsavg.pial_left, data_L, fsavg.sulc_left, "left", "lateral", vmax, titles[0])
        plot_brain(axs[1], fsavg.pial_left, data_L, fsavg.sulc_left, "left", "medial", vmax, titles[1])
        plot_brain(axs[2], fsavg.pial_right, data_R, fsavg.sulc_right, "right", "lateral", vmax, titles[2])
        plot_brain(axs[3], fsavg.pial_right, data_R, fsavg.sulc_right, "right", "medial", vmax, titles[3])

    add_colorbar(fig, fig.add_subplot(gs[0:3, 5]), vmax_main, "β")
    add_colorbar(fig, fig.add_subplot(gs[3, 5]), vmax_res, "residual β")

    fig.suptitle(
        f"{LABELS.get(metric, metric)}: empirical β map reconstructed from cortical eigenmodes\n"
        f"Full retained modes: r={r_full:.3f}, R²={r2_full:.3f}; "
        f"low-pass K={k_low}: r={r_low:.3f}, R²={r2_low:.3f}",
        y=0.985,
        fontsize=13,
    )

    stem = f"figure_empirical_full_lowpass_residual_{metric}_K{k_low}"
    png = OUT_FIG / f"{stem}.png"
    pdf = OUT_FIG / f"{stem}.pdf"

    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    out = pd.DataFrame([
        {
            "metric": metric,
            "k_low": k_low,
            "comparison": "empirical_vs_full_retained",
            "r": r_full,
            "r2": r2_full,
            "valid_vertices_L": int(mask_L.sum()),
            "valid_vertices_R": int(mask_R.sum()),
        },
        {
            "metric": metric,
            "k_low": k_low,
            "comparison": "empirical_vs_lowpass",
            "r": r_low,
            "r2": r2_low,
            "valid_vertices_L": int(mask_L.sum()),
            "valid_vertices_R": int(mask_R.sum()),
        },
    ])

    csv = OUT_TAB / f"{stem}_stats.csv"
    out.to_csv(csv, index=False)

    print(f"✅ wrote {png}")
    print(f"✅ wrote {pdf}")
    print(f"✅ wrote {csv}")
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="sentence_shift")
    ap.add_argument("--k-low", type=int, default=20)
    args = ap.parse_args()
    make_figure(args.metric, args.k_low)


if __name__ == "__main__":
    main()