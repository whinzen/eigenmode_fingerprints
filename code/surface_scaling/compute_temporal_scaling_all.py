# compute_temporal_scaling_all.py

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.signal import welch
import antropy as ant  # install via: pip install antropy

def compute_psd_and_fit(ts, fs=1.0):
    f, Pxx = welch(ts, fs=fs, nperseg=min(256, len(ts)))
    f = f[1:]
    Pxx = Pxx[1:]
    logf = np.log10(f)
    logP = np.log10(Pxx)
    slope, _ = np.polyfit(logf, logP, 1)
    return slope

def compute_scaling_metrics(X):
    T, V = X.shape
    slopes = []
    dfracts = []
    for v in range(V):
        ts = X[:, v]
        slope = compute_psd_and_fit(ts)
        dfrac = ant.petrosian_fd(ts)
        slopes.append(slope)
        dfracts.append(dfrac)
    return np.array(slopes), np.array(dfracts)

def main():
    BASE = Path.home() / "eigenmode_fingerprints"
    FUNC = BASE / "data/empirical"
    OUT = BASE / "pang_out" / "group"
    OUT.mkdir(parents=True, exist_ok=True)

    for subj_dir in sorted(FUNC.glob("sub-*")):
        subject = subj_dir.name
        for funcfile in sorted((subj_dir/"func").glob("*hemi-L*.func.gii")):
            try:
                run = funcfile.name.split("run-")[1].split("_")[0]
                func_r = funcfile.with_name(funcfile.name.replace("hemi-L", "hemi-R"))
                if not func_r.exists():
                    print(f"[WARN] No R hemisphere for {funcfile}")
                    continue

                X_L = np.stack([da.data for da in nib.load(str(funcfile)).darrays])
                X_R = np.stack([da.data for da in nib.load(str(func_r)).darrays])
                X = np.concatenate([X_L, X_R], axis=1)  # [T, V_L + V_R]

                slopes, dfracts = compute_scaling_metrics(X)
                df = pd.DataFrame({
                    "vertex": np.arange(len(slopes)),
                    "slope_alpha": slopes,
                    "fractal_dim": dfracts,
                    "subject": subject,
                    "run": run
                })
                outname = OUT / f"{subject}_run-{run}_temporal_scaling.csv"
                df.to_csv(outname, index=False)
                print(f"✅ Saved: {outname.name}")

            except Exception as e:
                print(f"[ERROR] {funcfile.name}: {e}")

if __name__ == "__main__":
    main()