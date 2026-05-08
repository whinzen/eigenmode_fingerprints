#!/usr/bin/env python

import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import nibabel as nib

from neuromaps.datasets import fetch_annotation
from neuromaps.transforms import fslr_to_fsaverage

# ---- runtime patch for neuromaps/nilearn path bug ----
import neuromaps.datasets.annotations as nma
from nilearn.datasets._utils import fetch_single_file as nilearn_fetch_single_file

def _fetch_file_pathsafe(url, data_dir, *args, **kwargs):
    return nilearn_fetch_single_file(url, Path(data_dir), *args, **kwargs)

nma._fetch_file = _fetch_file_pathsafe
# ------------------------------------------------------


BASE = Path.home() / "eigenmode_fingerprints"
EIG_DIR = BASE / "modes" / "fsaverage5"

L_FILE = EIG_DIR / "L_phi.npy"
R_FILE = EIG_DIR / "R_phi.npy"


def zscore(x):
    x = np.asarray(x, float).ravel()
    x = x - np.nanmean(x)
    s = np.nanstd(x)
    return x / s if s > 0 else x


def safe_corr(x, y):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()

    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    if len(x) < 3:
        return np.nan, np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    return pearsonr(x, y)


def gifti_to_array(gii):
    if isinstance(gii, (str, Path)):
        gii = nib.load(str(gii))
    return np.asarray(gii.darrays[0].data).ravel()


def fmt(x):
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x:.4f}"


def main():
    # ----------------------------
    # Load eigenmodes
    # ----------------------------
    L = np.load(L_FILE)
    R = np.load(R_FILE)

    mode3_L = zscore(L[:, 3])
    mode3_R = zscore(-R[:, 3])   # same RH sign convention as before

    # ----------------------------
    # Fetch canonical Margulies Gradient 1
    # ----------------------------
    grad = fetch_annotation(
        source="margulies2016",
        desc="fcgradient01",
        space="fsLR",
        den="32k",
        return_single=True,
    )

    # resample fsLR 32k -> fsaverage 10k
    grad_fsavg = fslr_to_fsaverage(grad, target_density="10k", method="linear")

    grad_L = zscore(gifti_to_array(grad_fsavg[0]))
    grad_R = zscore(gifti_to_array(grad_fsavg[1]))

    print("Shapes:")
    print(" mode3_L:", mode3_L.shape, "grad_L:", grad_L.shape)
    print(" mode3_R:", mode3_R.shape, "grad_R:", grad_R.shape)

    rL, pL = safe_corr(mode3_L, grad_L)
    rR, pR = safe_corr(mode3_R, grad_R)

    print("\n=== Mode 3 vs canonical Margulies Gradient 1 ===")
    print(f"Left : r = {fmt(rL)}, |r| = {fmt(abs(rL))}, p = {pL:.3e}")
    print(f"Right: r = {fmt(rR)}, |r| = {fmt(abs(rR))}, p = {pR:.3e}")

    out = BASE / "pang_out" / "paper_tables" / "table_mode3_vs_margulies_gradient1.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("hemi,r,abs_r,p\n")
        f.write(f"L,{rL},{abs(rL)},{pL}\n")
        f.write(f"R,{rR},{abs(rR)},{pR}\n")

    print(f"\n✅ wrote {out}")


if __name__ == "__main__":
    main()