#!/usr/bin/env python

import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from nilearn import datasets, image, surface


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
        raise ValueError(f"Shape mismatch in correlation: x={x.shape}, y={y.shape}")

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    if len(x) < 3:
        return np.nan, np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan

    return pearsonr(x, y)


def fmt(x):
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x:.4f}"


def main():
    # ----------------------------
    # Load eigenmodes
    # ----------------------------
    L = np.load(L_FILE)   # [V x K]
    R = np.load(R_FILE)

    # Mode 3; same RH display convention as before
    mode3_L = L[:, 3].ravel()
    mode3_R = (-R[:, 3]).ravel()

    print("Mode 3 shapes:", mode3_L.shape, mode3_R.shape)

    # ----------------------------
    # Fetch fsaverage5 and Yeo atlas
    # ----------------------------
    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    yeo = datasets.fetch_atlas_yeo_2011(n_networks=7, thickness="thick")

    atlas_img = image.load_img(yeo["maps"])
    print("Yeo labels:", yeo["labels"])

    # ----------------------------
    # Project volumetric atlas to surface
    # ----------------------------
    atlas_L = surface.vol_to_surf(
        atlas_img,
        fsavg.pial_left,
        interpolation="nearest_most_frequent",
    )
    atlas_R = surface.vol_to_surf(
        atlas_img,
        fsavg.pial_right,
        interpolation="nearest_most_frequent",
    )

    atlas_L = np.asarray(atlas_L).ravel().astype(int)
    atlas_R = np.asarray(atlas_R).ravel().astype(int)

    print("Atlas shapes:", atlas_L.shape, atlas_R.shape)
    print("Unique labels L:", np.unique(atlas_L))
    print("Unique labels R:", np.unique(atlas_R))

    # ----------------------------
    # Yeo 7 labels
    # 0 background
    # 1 Visual
    # 2 Somatomotor
    # 3 Dorsal Attention
    # 4 Ventral Attention / Salience
    # 5 Limbic
    # 6 Frontoparietal
    # 7 Default Mode
    # ----------------------------
    DMN_LABEL = 7

    dmn_L = (atlas_L == DMN_LABEL).astype(float)
    dmn_R = (atlas_R == DMN_LABEL).astype(float)

    print("DMN vertices L:", int(dmn_L.sum()))
    print("DMN vertices R:", int(dmn_R.sum()))

    # ----------------------------
    # Correlate with mode 3
    # ----------------------------
    mode3_Lz = zscore(mode3_L)
    mode3_Rz = zscore(mode3_R)
    dmn_Lz = zscore(dmn_L)
    dmn_Rz = zscore(dmn_R)

    print("Z-scored shapes:", mode3_Lz.shape, mode3_Rz.shape, dmn_Lz.shape, dmn_Rz.shape)

    rL, pL = safe_corr(mode3_Lz, dmn_Lz)
    rR, pR = safe_corr(mode3_Rz, dmn_Rz)

    print("\n=== Mode 3 vs DMN (Yeo 7) ===")
    print(f"Left : r = {fmt(rL)}, |r| = {fmt(abs(rL) if np.isfinite(rL) else np.nan)}, p = {pL}")
    print(f"Right: r = {fmt(rR)}, |r| = {fmt(abs(rR) if np.isfinite(rR) else np.nan)}, p = {pR}")

    # ----------------------------
    # Save table
    # ----------------------------
    out_dir = BASE / "pang_out" / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "table_mode3_vs_dmn.csv"

    with open(out_csv, "w") as f:
        f.write("hemi,r,abs_r,p\n")
        f.write(f"L,{rL},{abs(rL) if np.isfinite(rL) else np.nan},{pL}\n")
        f.write(f"R,{rR},{abs(rR) if np.isfinite(rR) else np.nan},{pR}\n")

    print(f"\n✅ wrote {out_csv}")


if __name__ == "__main__":
    main()