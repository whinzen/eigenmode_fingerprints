# ~/eigenmode_fingerprints/code/energy_per_run_timeseries.py

from pathlib import Path
import numpy as np
import nibabel as nib

BASE = Path.home() / "eigenmode_fingerprints"
EMP  = BASE / "data" / "empirical"
EIG  = BASE / "modes/fsaverage5"
OUT  = BASE / "pang_out" / "energy"  # <--- this is the missing folder

PHI_L = np.load(EIG / "phi_L.npy")
PHI_R = np.load(EIG / "phi_R.npy")
LAM_L = np.load(EIG / "lam_L.npy")  # Not used here, but could be
LAM_R = np.load(EIG / "lam_R.npy")

def load_gii(path):
    gii = nib.load(str(path))
    d = np.stack([arr.data for arr in gii.darrays])  # [T, V]
    return d.T  # → [V, T]

def zscore(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

def save_energy_timeseries(hemi, run, A2):
    out_dir = OUT / hemi
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run-{run}.npy"
    np.save(out_path, A2)
    print(f"✅ Saved: {out_path} [shape {A2.shape}]")

def main():
    for subj_dir in sorted(EMP.glob("sub-*")):
        func = subj_dir / "func"
        if not func.exists():
            continue
        for hemi, PHI in [("L", PHI_L), ("R", PHI_R)]:
            for f in sorted(func.glob(f"*run-*_hemi-{hemi}_space-fsaverage5_bold.func.gii")):
                run = f.name.split("_run-")[1].split("_")[0]
                try:
                    X = zscore(load_gii(f))  # [V, T]
                    A = PHI.T @ X             # [K, T]
                    A2 = A ** 2               # [K, T]
                    save_energy_timeseries(hemi, run, A2)
                except Exception as e:
                    print(f"[WARN] Failed on {f.name}: {e}")

if __name__ == "__main__":
    main()