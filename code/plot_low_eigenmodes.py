import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from nilearn import plotting
from nilearn.datasets import fetch_surf_fsaverage


BASE = Path.home() / "eigenmode_fingerprints"
EIG_DIR = BASE / "modes" / "fsaverage5"

L_FILE = EIG_DIR / "L_phi.npy"
R_FILE = EIG_DIR / "R_phi.npy"


def load_modes():
    L = np.load(L_FILE)  # [V x K]
    R = np.load(R_FILE)
    return L, R


def align_mode_signs(L, R, modes):
    """
    Align right hemisphere signs to left hemisphere
    """
    R_aligned = R.copy()
    for k in modes:
        l = L[:, k] / np.linalg.norm(L[:, k])
        r = R[:, k] / np.linalg.norm(R[:, k])
        if np.dot(l, r) < 0:
            R_aligned[:, k] *= -1
    return R_aligned


def plot_modes_grid(modes=[1, 2, 3, 4, 5, 6]):
    """
    Plot eigenmodes in compact grid
    """

    # ----------------------------
    # Load + align signs
    # ----------------------------
    L, R = load_modes()
    R = align_mode_signs(L, R, modes)

    fsavg = fetch_surf_fsaverage(mesh="fsaverage5")

    n = len(modes)

    # More compact figure
    fig = plt.figure(figsize=(8, 1.8 * n))
    fig.patch.set_facecolor("white")

    for i, k in enumerate(modes):

        mode_L = L[:, k]
        mode_R = R[:, k]

        # Normalize per mode
        mode_L = mode_L / np.max(np.abs(mode_L))
        mode_R = mode_R / np.max(np.abs(mode_R))

        # LEFT
        ax = plt.subplot(n, 2, 2*i + 1, projection="3d")
        plotting.plot_surf_stat_map(
            fsavg.infl_left,
            mode_L,
            hemi="left",
            view="lateral",
            cmap="coolwarm",
            colorbar=False,
            axes=ax,
            title=f"Mode {k}"
        )

        # RIGHT
        ax = plt.subplot(n, 2, 2*i + 2, projection="3d")
        plotting.plot_surf_stat_map(
            fsavg.infl_right,
            mode_R,
            hemi="right",
            view="lateral",
            cmap="coolwarm",
            colorbar=False,
            axes=ax,
            title=""  # avoid duplication → saves space
        )

    # Tighten spacing aggressively
    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.98,
        bottom=0.02,
        wspace=0.02,
        hspace=0.15
    )

    out_png = BASE / "pang_out" / "paper_figures" / "figure_low_modes.png"
    out_pdf = BASE / "pang_out" / "paper_figures" / "figure_low_modes.pdf"

    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)

    print(f"✅ wrote {out_png}")
    print(f"✅ wrote {out_pdf}")

    plt.show()


if __name__ == "__main__":
    plot_modes_grid([1, 2, 3, 4, 5, 6])