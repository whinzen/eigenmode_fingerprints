#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd

def load_mgh_ts(path: Path) -> np.ndarray:
    """Return array (V, T) from fsaverage5 .mgh/.mgz surface time series."""
    img = nib.load(str(path))
    data = img.get_fdata()  # expected (V, 1, 1, T)
    if data.ndim != 4 or data.shape[1:3] != (1,1):
        raise ValueError(f"Unexpected shape {data.shape} for {path}")
    V, _, _, T = data.shape
    return data.reshape(V, T)

def zscore_time(VT: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score each vertex time series: (V,T) -> (V,T)."""
    mu = VT.mean(axis=1, keepdims=True)
    sd = VT.std(axis=1, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (VT - mu) / sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lh", required=True, help="lh .mgh/.mgz (fsaverage5)")
    ap.add_argument("--rh", required=True, help="rh .mgh/.mgz (fsaverage5)")
    ap.add_argument("--modes", required=True, help="dir with phi_*.npy, lam_*.npy (fsaverage5)")
    ap.add_argument("--outdir", required=True, help="output dir")
    ap.add_argument("--run", required=True, help="label for outputs (e.g., run15)")
    ap.add_argument("--K", type=int, default=120, help="number of modes to use")
    ap.add_argument("--TR", type=float, default=2.0, help="seconds (only for reporting)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load surface data
    YL = load_mgh_ts(Path(args.lh))   # (VL, T)
    YR = load_mgh_ts(Path(args.rh))   # (VR, T)
    T = YL.shape[1]
    print(f"Loaded: LH {YL.shape}  RH {YR.shape}  (T={T}, TR={args.TR}s)")

    # 2) Z-score per vertex
    ZL = zscore_time(YL)
    ZR = zscore_time(YR)

    # 3) Load eigenmodes (phi: (V,K0), lam: (K0,))
    modes_dir = Path(args.modes) / "fsaverage5"
    phi_L = np.load(modes_dir / "phi_L.npy")   # (VL, K0)
    phi_R = np.load(modes_dir / "phi_R.npy")   # (VR, K0)
    lam_L = np.load(modes_dir / "lam_L.npy")   # (K0,)
    lam_R = np.load(modes_dir / "lam_R.npy")   # (K0,)

    # 4) Truncate to K (and sanity-check V matches)
    K = min(args.K, phi_L.shape[1], phi_R.shape[1])
    if phi_L.shape[0] != ZL.shape[0] or phi_R.shape[0] != ZR.shape[0]:
        raise ValueError(f"Vertex count mismatch: "
                         f"L phi {phi_L.shape[0]} vs data {ZL.shape[0]}, "
                         f"R phi {phi_R.shape[0]} vs data {ZR.shape[0]}")
    phi_L = phi_L[:, :K]
    phi_R = phi_R[:, :K]
    lam_L = lam_L[:K]
    lam_R = lam_R[:K]

    # 5) Project: A = phi^T * Z  → shape (K, T)
    A_L = phi_L.T @ ZL
    A_R = phi_R.T @ ZR

    # 6) Energy per mode
    E_L = (A_L**2).mean(axis=1)  # (K,)
    E_R = (A_R**2).mean(axis=1)  # (K,)
    print(f"Energy stats (LH): min={E_L.min():.3g} max={E_L.max():.3g} nonzero={(E_L>0).sum()}/{K}")
    print(f"Energy stats (RH): min={E_R.min():.3g} max={E_R.max():.3g} nonzero={(E_R>0).sum()}/{K}")

    # 7) Save summary CSVs
    dfL = pd.DataFrame({"k": np.arange(1, K+1), "lambda": lam_L, "E": E_L, "hemi": "L", "run": args.run})
    dfR = pd.DataFrame({"k": np.arange(1, K+1), "lambda": lam_R, "E": E_R, "hemi": "R", "run": args.run})
    df  = pd.concat([dfL, dfR], ignore_index=True)
    out_csv = outdir / f"{args.run}_mode_energy_fs5.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

    # 8) Also save the time series in mode space (optional, large)
    np.save(outdir / f"{args.run}_A_L.npy", A_L.astype(np.float32))
    np.save(outdir / f"{args.run}_A_R.npy", A_R.astype(np.float32))
    np.save(outdir / f"{args.run}_lambda_L.npy", lam_L.astype(np.float64))
    np.save(outdir / f"{args.run}_lambda_R.npy", lam_R.astype(np.float64))
    print("Saved A_L/A_R and lambda arrays.")

if __name__ == "__main__":
    main()