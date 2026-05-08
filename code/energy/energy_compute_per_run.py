#!/usr/bin/env python
"""
energy_compute_per_run.py

Compute modal energies E_k per run and hemisphere:
  - loads eigenmodes (phi, lam) for fsaverage5
  - loads surface BOLD (func.gii) per run/hemisphere
  - z-score per-vertex time series
  - project: A = phi^T X
  - energy: E_k = mean_t(A_k^2)
Saves per-run CSVs under pang_out/sub-XX/run-YY:
  energy_{hemi}.csv with columns [mode_k, lam, E]
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

# --------- CONFIG ---------
BASE = Path.home()/"eigenmode_fingerprints"
EMP  = BASE/"data/empirical"
OUT  = BASE/"pang_out"
EIG  = BASE/"modes/fsaverage5"

PHI_L = EIG/"phi_L.npy"
PHI_R = EIG/"phi_R.npy"
LAM_L = EIG/"lam_L.npy"
LAM_R = EIG/"lam_R.npy"
# -------------------------

def load_modes():
    phiL = np.load(PHI_L)
    phiR = np.load(PHI_R)
    lamL = np.load(LAM_L)
    lamR = np.load(LAM_R)
    return (phiL, lamL), (phiR, lamR)

def zscore_time(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd==0] = 1.0
    return (X - mu)/sd

def project_energy(phi, Xz):
    A = phi.T @ Xz
    E = (A**2).mean(axis=1)
    return E

def load_gii_timeseries(path_gii):
    g = nib.load(str(path_gii))
    d = np.array([da.data for da in g.darrays])
    X = d.T
    return X

def process_subject(sub_dir, phiL, lamL, phiR, lamR):
    func_dir = sub_dir/"func"
    if not func_dir.exists():
        return []

    rows = []
    for hemi in ["L", "R"]:
        for f in sorted(func_dir.glob(f"*run-*_hemi-{hemi}_space-fsaverage5_bold.func.gii")):
            run = f.name.split("_run-")[1].split("_")[0]
            try:
                X = load_gii_timeseries(f)
                Xz = zscore_time(X)
                if hemi == "L":
                    E = project_energy(phiL, Xz)
                    lam = lamL
                else:
                    E = project_energy(phiR, Xz)
                    lam = lamR
                k = np.arange(len(E))
                df = pd.DataFrame({"mode_k": k, "lam": lam, "E": E})
                out_dir = OUT/sub_dir.name/f"run-{run}"
                out_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_dir/f"energy_{hemi}.csv", index=False)
                rows.append((sub_dir.name, run, hemi, len(E)))
                print(f"[{sub_dir.name} run-{run} hemi-{hemi}] OK | K={len(E)} | out={out_dir}")
            except Exception as e:
                print(f"[WARN] Failed {f}: {e}")
    return rows

def main():
    (phiL, lamL), (phiR, lamR) = load_modes()
    EMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    coverage = []
    for sdir in sorted(EMP.glob("sub-*")):
        coverage += process_subject(sdir, phiL, lamL, phiR, lamR)

    cov_df = pd.DataFrame(coverage, columns=["subject", "run", "hemi", "K"])
    cov_df.to_csv(OUT/"group"/"run_coverage_report.tsv", sep="\t", index=False)
    print("Wrote:", OUT/"group"/"run_coverage_report.tsv")

if __name__ == "__main__":
    main()
