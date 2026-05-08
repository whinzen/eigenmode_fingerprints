import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from nilearn import plotting
from nilearn.datasets import fetch_surf_fsaverage

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

EIG_DIR = BASE / "modes" / "fsaverage5"
L_FILE = EIG_DIR / "L_phi.npy"
R_FILE = EIG_DIR / "R_phi.npy"

fsavg = fetch_surf_fsaverage(mesh="fsaverage5")


def load_modes():
    L = np.load(L_FILE)   # [V x K]
    R = np.load(R_FILE)   # [V x K]
    return L, R


def align_mode_signs(L, R, max_k=None):
    R_aligned = R.copy()
    K = L.shape[1] if max_k is None else min(max_k, L.shape[1])
    for k in range(K):
        l = L[:, k] / np.linalg.norm(L[:, k])
        r = R[:, k] / np.linalg.norm(R[:, k])
        if np.dot(l, r) < 0:
            R_aligned[:, k] *= -1
    return R_aligned


def load_beta_profile_singlefile(csv_file):
    """
    For files with columns including hemi, mode_k, beta_mean
    """
    df = pd.read_csv(csv_file)

    L = df[df["hemi"] == "L"].copy()
    R = df[df["hemi"] == "R"].copy()

    beta_L = L.sort_values("mode_k")["beta_mean"].values
    beta_R = R.sort_values("mode_k")["beta_mean"].values

    return beta_L, beta_R


def load_beta_profile_lr(csv_L, csv_R):
    """
    For files split by hemisphere
    """
    L = pd.read_csv(csv_L).copy()
    R = pd.read_csv(csv_R).copy()

    if "k" in L.columns and "mode_k" not in L.columns:
        L = L.rename(columns={"k": "mode_k"})
    if "k" in R.columns and "mode_k" not in R.columns:
        R = R.rename(columns={"k": "mode_k"})

    beta_L = L.sort_values("mode_k")["beta_mean"].values
    beta_R = R.sort_values("mode_k")["beta_mean"].values

    return beta_L, beta_R


def reconstruct_map(phi, beta, kmax=None, drop_mode0=True):
    start = 1 if drop_mode0 else 0
    end = len(beta) if kmax is None else min(kmax, len(beta))
    return phi[:, start:end] @ beta[start:end]


def normalize_map(x):
    m = np.max(np.abs(x))
    return x / m if m > 0 else x


def main():
    # Example: original curvature split by hemisphere
    csv_L = PANG / "group_curvature_glm" / "group_curvature_hemi-L_by_mode_subject_level.csv"
    csv_R = PANG / "group_curvature_glm" / "group_curvature_hemi-R_by_mode_subject_level.csv"

    beta_L, beta_R = load_beta_profile_lr(csv_L, csv_R)

    L, R = load_modes()
    R = align_mode_signs(L, R, max_k=len(beta_L))

    # reconstruct using first 20 non-constant modes
    recon_L = reconstruct_map(L, beta_L, kmax=20, drop_mode0=True)
    recon_R = reconstruct_map(R, beta_R, kmax=20, drop_mode0=True)

    recon_L = normalize_map(recon_L)
    recon_R = normalize_map(recon_R)

    fig = plt.figure(figsize=(10, 4))
    fig.patch.set_facecolor("white")

    ax1 = plt.subplot(1, 2, 1, projection="3d")
    plotting.plot_surf_stat_map(
        fsavg.infl_left,
        recon_L,
        hemi="left",
        view="lateral",
        cmap="coolwarm",
        colorbar=True,
        axes=ax1,
        title="Reconstructed beta map (L)"
    )

    ax2 = plt.subplot(1, 2, 2, projection="3d")
    plotting.plot_surf_stat_map(
        fsavg.infl_right,
        recon_R,
        hemi="right",
        view="lateral",
        cmap="coolwarm",
        colorbar=True,
        axes=ax2,
        title="Reconstructed beta map (R)"
    )

    plt.tight_layout()

    out_png = PANG / "paper_figures" / "figure_reconstructed_beta_map.png"
    out_pdf = PANG / "paper_figures" / "figure_reconstructed_beta_map.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    print(f"✅ wrote {out_png}")
    print(f"✅ wrote {out_pdf}")


if __name__ == "__main__":
    main()