#!/usr/bin/env python
"""
energy_group_aggregate.py

Collect per-run energies saved by energy_compute_per_run.py,
aggregate to per-subject (mean over runs, both hemispheres), then
group (mean & SEM across subjects). Save:
  - energy_spectrum_group.csv  [k, lam, Emean, Esem]
  - Emean_group.npy, Esem_group.npy, lam_group.npy
  - group_energy_vs_lambda.png, group_energy_vs_index.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home()/"eigenmode_fingerprints"
OUT  = BASE/"pang_out"
GRP  = OUT/"group"
GRP.mkdir(parents=True, exist_ok=True)

def gather_rows():
    rows = []
    for sub_dir in sorted(OUT.glob("sub-*")):
        for run_dir in sorted(sub_dir.glob("run-*")):
            for hemi in ["L","R"]:
                f = run_dir/f"energy_{hemi}.csv"
                if f.exists():
                    df = pd.read_csv(f)
                    df["subject"] = sub_dir.name
                    df["run"] = run_dir.name.split("-")[1]
                    df["hemi"] = hemi
                    rows.append(df)
    if not rows:
        raise SystemExit("No per-run energy CSVs found. Run energy_compute_per_run.py first.")
    return pd.concat(rows, ignore_index=True)

def agg_subject_level(df):
    # average across runs and hemispheres within each subject per mode
    sub = (df.groupby(["subject","mode_k"], as_index=False)
             .agg(lam=("lam","first"), Emean=("E","mean")))
    return sub

def agg_group_level(sub):
    g = (sub.groupby(["mode_k"], as_index=False)
            .agg(lam=("lam","first"),
                 Emean=("Emean","mean"),
                 Esem=("Emean", lambda x: x.std(ddof=1)/np.sqrt(len(x)))))
    return g

def save_arrays(g):
    np.save(GRP/"lam_group.npy", g["lam"].to_numpy())
    np.save(GRP/"Emean_group.npy", g["Emean"].to_numpy())
    np.save(GRP/"Esem_group.npy", g["Esem"].to_numpy())

def plot_group(g):
    # by index
    plt.figure(figsize=(6.5,3.8))
    x = g["mode_k"].to_numpy()
    y = g["Emean"].to_numpy()
    e = g["Esem"].to_numpy()
    plt.plot(x, y, lw=2)
    plt.fill_between(x, y-e, y+e, alpha=0.2, linewidth=0)
    plt.xlabel("Eigenmode index k")
    plt.ylabel("Energy E_k")
    plt.title("Group energy spectrum (by mode index)")
    plt.tight_layout()
    plt.savefig(GRP/"group_energy_vs_index.png", dpi=150)
    plt.close()

    # by lambda (log-log)
    plt.figure(figsize=(6.5,3.8))
    x = g["lam"].to_numpy()
    y = g["Emean"].to_numpy()
    e = g["Esem"].to_numpy()
    plt.loglog(x, y, lw=2)
    # draw symmetric CI on log scale by plotting bands in data coords
    plt.fill_between(x, np.clip(y-e, 1e-12, None), y+e, alpha=0.15, linewidth=0)
    plt.xlabel("Eigenvalue λ (log)")
    plt.ylabel("Energy E_k (log)")
    plt.title("Group energy spectrum (log–log)")
    plt.tight_layout()
    plt.savefig(GRP/"group_energy_vs_lambda.png", dpi=150)
    plt.close()

def main():
    df = gather_rows()
    sub = agg_subject_level(df)
    g = agg_group_level(sub)
    g = g.sort_values("mode_k")
    g.to_csv(GRP/"energy_spectrum_group.csv", index=False)
    save_arrays(g)
    plot_group(g)
    print("Wrote:")
    print(" ", GRP/"energy_spectrum_group.csv")
    print(" ", GRP/"group_energy_vs_index.png")
    print(" ", GRP/"group_energy_vs_lambda.png")
    print(" ", GRP/"Emean_group.npy")
    print(" ", GRP/"Esem_group.npy")
    print(" ", GRP/"lam_group.npy")

if __name__=="__main__":
    main()