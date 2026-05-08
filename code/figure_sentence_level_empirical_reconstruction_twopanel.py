#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from nilearn import datasets, plotting

BASE = Path.home() / "eigenmode_fingerprints"
MODE_DIR = BASE / "modes" / "fsaverage5"
BETA_DIR = BASE / "pang_out" / "vertex_betas"
OUT_FIG = BASE / "pang_out" / "paper_figures"
OUT_TAB = BASE / "pang_out" / "paper_tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

K_LOW = 20

METRICS = [
    ("boundary", "Sentence boundary"),
    ("sentence_shift", "Sentence shift"),
]


def load_phi(hemi):
    return np.load(MODE_DIR / f"{hemi}_phi.npy")


def load_beta(metric, hemi):
    f = BETA_DIR / f"{metric}_{hemi}.npy"
    if not f.exists():
        raise FileNotFoundError(f)
    x = np.load(f).astype(float)
    x[x == 0] = np.nan
    return x


def reconstruct(beta_map, phi, k_low):
    beta_map = np.asarray(beta_map, float)
    mask = np.isfinite(beta_map)

    y = np.full_like(beta_map, np.nan, dtype=float)
    y[mask] = beta_map[mask] - np.nanmean(beta_map[mask])

    B_full = phi[:, 1:]
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

    r = np.corrcoef(x, y)[0, 1]
    ss_res = np.sum((x - y) ** 2)
    ss_tot = np.sum((x - x.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return r, r2


def robust_vmax(arrays, pct=99):
    vals = []
    for a in arrays:
        vals.append(a[np.isfinite(a)])
    x = np.concatenate(vals)
    return np.nanpercentile(np.abs(x), pct)


def plot_brain(ax, surf, data, bg, hemi, view, vmax):
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
        title="",
    )


def add_colorbar(fig, cax, vmax, label):
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def compute_metric(metric):
    phi_L = load_phi("L")
    phi_R = load_phi("R")

    emp_L_raw = load_beta(metric, "L")
    emp_R_raw = load_beta(metric, "R")

    emp_L, full_L, low_L, res_L, mask_L = reconstruct(emp_L_raw, phi_L, K_LOW)
    emp_R, full_R, low_R, res_R, mask_R = reconstruct(emp_R_raw, phi_R, K_LOW)

    emp = np.concatenate([emp_L, emp_R])
    full = np.concatenate([full_L, full_R])
    low = np.concatenate([low_L, low_R])

    r_full, r2_full = stats(emp, full)
    r_low, r2_low = stats(emp, low)

    vmax_main = robust_vmax([emp_L, emp_R, full_L, full_R, low_L, low_R], pct=99)
    vmax_res = robust_vmax([res_L, res_R], pct=99)

    return {
        "emp_L": emp_L, "emp_R": emp_R,
        "full_L": full_L, "full_R": full_R,
        "low_L": low_L, "low_R": low_R,
        "res_L": res_L, "res_R": res_R,
        "r_full": r_full, "r2_full": r2_full,
        "r_low": r_low, "r2_low": r2_low,
        "valid_L": int(mask_L.sum()),
        "valid_R": int(mask_R.sum()),
        "vmax_main": vmax_main,
        "vmax_res": vmax_res,
    }


def main():
    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    data = {metric: compute_metric(metric) for metric, _ in METRICS}

    row_specs = [
        ("A. Empirical\nvertexwise β map\n(demeaned)", "emp"),
        ("B. Full retained-mode\nreconstruction", "full"),
        (f"C. Low-pass reconstruction\nmodes 1–{K_LOW}", "low"),
        ("D. Residual\nempirical − low-pass", "res"),
    ]

    fig = plt.figure(figsize=(18.5, 10.8), facecolor="white")
    gs = fig.add_gridspec(
        nrows=4,
        ncols=12,
        width_ratios=[1.35, 1, 1, 1, 1, 0.075, 0.22, 1, 1, 1, 1, 0.075],
        wspace=0.02,
        hspace=0.02,
    )

    fig.suptitle(
        "Sentence-level empirical β maps are reconstructed by low-order cortical eigenmodes",
        y=0.995,
        fontsize=14,
    )

    # Panel headers lower than title
    fig.text(0.285, 0.955, "Sentence boundary", ha="center", va="bottom",
             fontsize=13, fontweight="bold")
    fig.text(0.745, 0.955, "Sentence shift", ha="center", va="bottom",
             fontsize=13, fontweight="bold")

    # View headers lower still
    view_titles = ["L lateral", "L medial", "R lateral", "R medial"]
    x_left = [0.188, 0.266, 0.344, 0.422]
    x_right = [0.650, 0.728, 0.806, 0.884]
    for x, lab in zip(x_left, view_titles):
        fig.text(x, 0.925, lab, ha="center", va="bottom", fontsize=9)
    for x, lab in zip(x_right, view_titles):
        fig.text(x, 0.925, lab, ha="center", va="bottom", fontsize=9)

    stats_rows = []

    for row_i, (row_label, key) in enumerate(row_specs):
        lab_ax = fig.add_subplot(gs[row_i, 0])
        lab_ax.axis("off")
        lab_ax.text(
            0.98, 0.38, row_label,
            ha="right", va="center",
            fontsize=11, fontweight="bold",
        )

        for panel_i, (metric, label) in enumerate(METRICS):
            d = data[metric]
            col_offset = 1 if panel_i == 0 else 7

            data_L = d[f"{key}_L"]
            data_R = d[f"{key}_R"]
            vmax = d["vmax_res"] if key == "res" else d["vmax_main"]

            axes = [
                fig.add_subplot(gs[row_i, col_offset + 0], projection="3d"),
                fig.add_subplot(gs[row_i, col_offset + 1], projection="3d"),
                fig.add_subplot(gs[row_i, col_offset + 2], projection="3d"),
                fig.add_subplot(gs[row_i, col_offset + 3], projection="3d"),
            ]

            plot_brain(axes[0], fsavg.pial_left, data_L, fsavg.sulc_left, "left", "lateral", vmax)
            plot_brain(axes[1], fsavg.pial_left, data_L, fsavg.sulc_left, "left", "medial", vmax)
            plot_brain(axes[2], fsavg.pial_right, data_R, fsavg.sulc_right, "right", "lateral", vmax)
            plot_brain(axes[3], fsavg.pial_right, data_R, fsavg.sulc_right, "right", "medial", vmax)

    # Non-overlapping colorbars: one pair per panel
    boundary = data["boundary"]
    shift = data["sentence_shift"]

    add_colorbar(fig, fig.add_subplot(gs[0:3, 5]), boundary["vmax_main"], "β")
    add_colorbar(fig, fig.add_subplot(gs[3, 5]), boundary["vmax_res"], "resid.")

    add_colorbar(fig, fig.add_subplot(gs[0:3, 11]), shift["vmax_main"], "β")
    add_colorbar(fig, fig.add_subplot(gs[3, 11]), shift["vmax_res"], "resid.")

    # Stats text below headers
    for x, (metric, _) in zip([0.285, 0.745], METRICS):
        d = data[metric]
        fig.text(
            x, 0.905,
            f"Full: r={d['r_full']:.3f}, R²={d['r2_full']:.3f}; "
            f"K={K_LOW}: r={d['r_low']:.3f}, R²={d['r2_low']:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

        stats_rows.append({
            "metric": metric,
            "k_low": K_LOW,
            "full_r": d["r_full"],
            "full_r2": d["r2_full"],
            "lowpass_r": d["r_low"],
            "lowpass_r2": d["r2_low"],
            "valid_vertices_L": d["valid_L"],
            "valid_vertices_R": d["valid_R"],
        })

    out_png = OUT_FIG / f"figure_sentence_level_empirical_reconstruction_twopanel_K{K_LOW}.png"
    out_pdf = OUT_FIG / f"figure_sentence_level_empirical_reconstruction_twopanel_K{K_LOW}.pdf"
    out_csv = OUT_TAB / f"figure_sentence_level_empirical_reconstruction_twopanel_K{K_LOW}_stats.csv"

    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(out_pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(stats_rows).to_csv(out_csv, index=False)

    print(f"✅ wrote {out_png}")
    print(f"✅ wrote {out_pdf}")
    print(f"✅ wrote {out_csv}")
    print(pd.DataFrame(stats_rows))


if __name__ == "__main__":
    main()