#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"
EIG_DIR = BASE / "modes" / "fsaverage5"

OUT_DIR = PANG / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KMAX = 20

# ---------- paths ----------
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

# ---------- load eigenmodes ----------
phi_L = np.load(EIG_DIR / "L_phi.npy")
phi_R = np.load(EIG_DIR / "R_phi.npy")


def load_beta_profile_from_single(csv_file: Path):
    df = pd.read_csv(csv_file).copy()

    if "k" in df.columns and "mode_k" not in df.columns:
        df = df.rename(columns={"k": "mode_k"})

    beta_L = df[df["hemi"].isin(["L", "hemi-L"])].sort_values("mode_k")["beta_mean"].values
    beta_R = df[df["hemi"].isin(["R", "hemi-R"])].sort_values("mode_k")["beta_mean"].values

    return beta_L, beta_R


def load_beta_profile_from_pair(csv_L: Path, csv_R: Path):
    L = pd.read_csv(csv_L)
    R = pd.read_csv(csv_R)

    if "k" in L.columns and "mode_k" not in L.columns:
        L = L.rename(columns={"k": "mode_k"})
    if "k" in R.columns and "mode_k" not in R.columns:
        R = R.rename(columns={"k": "mode_k"})

    beta_L = L.sort_values("mode_k")["beta_mean"].values
    beta_R = R.sort_values("mode_k")["beta_mean"].values

    return beta_L, beta_R


def load_beta_pair(spec):
    if "single" in spec:
        return load_beta_profile_from_single(spec["single"])
    return load_beta_profile_from_pair(spec["L"], spec["R"])


def reconstruct(phi, beta):
    kmax = min(KMAX, phi.shape[1] - 1, len(beta) - 1)
    return phi[:, 1:kmax + 1] @ beta[1:kmax + 1]


# ---------- build reconstructed maps ----------
maps = {}
for name, spec in REGRESSORS.items():
    beta_L, beta_R = load_beta_pair(spec)

    recon_L = reconstruct(phi_L, beta_L)
    recon_R = reconstruct(phi_R, beta_R)

    # concatenate hemispheres
    full = np.concatenate([recon_L, recon_R])

    # normalize for shape comparison
    full = full / np.max(np.abs(full))

    maps[name] = full


# ---------- correlation matrix ----------
names = list(maps.keys())
mat = np.zeros((len(names), len(names)))

for i, n1 in enumerate(names):
    for j, n2 in enumerate(names):
        x = maps[n1]
        y = maps[n2]

        r = np.corrcoef(x, y)[0, 1]
        mat[i, j] = r


# ---------- save CSV ----------
df = pd.DataFrame(mat, index=names, columns=names)
csv_out = OUT_DIR / "table_reconstructed_map_correlations.csv"
df.to_csv(csv_out)

print("\nCorrelation matrix:")
print(df.round(3))
print(f"\n✅ wrote {csv_out}")


# ---------- plot heatmap ----------
fig, ax = plt.subplots(figsize=(6.5, 5.5))

im = ax.imshow(mat, vmin=0.9, vmax=1.0, cmap="magma_r")

ax.set_xticks(range(len(names)))
ax.set_yticks(range(len(names)))
ax.set_xticklabels(names, rotation=35, ha="right")
ax.set_yticklabels(names)

for i in range(len(names)):
    for j in range(len(names)):
        val = mat[i, j]
        color = "white" if val > 0.96 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.outline.set_visible(False)

ax.set_title("Correlation of reconstructed eigenmode maps", pad=12)

fig.tight_layout()

png = OUT_DIR / "figure_reconstructed_map_correlations.png"
pdf = OUT_DIR / "figure_reconstructed_map_correlations.pdf"

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")

plt.close(fig)

print(f"✅ wrote {png}")
print(f"✅ wrote {pdf}")