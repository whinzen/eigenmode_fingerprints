#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from settings import PANG_OUT


BASE = Path.home() / "eigenmode_fingerprints"
PANG = Path(PANG_OUT)

K_VALUES = [1, 2, 3, 5, 10, 20, 40, 60, 80, 100, 120]

OUT_DIR = PANG / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_DIR = PANG / "paper_tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def cumulative_energy_fraction(A, k):
    """
    Fast cumulative variance/energy estimate within retained eigenmode subspace.

    A has shape [K x T], where rows are eigenmode coefficient time series.
    This computes:

        sum_{i < k} A_i^2 / sum_{i < Kmax} A_i^2

    This is the cumulative fraction of retained eigenmode energy, not full
    vertexwise BOLD variance.
    """
    A = np.asarray(A, float)

    if A.ndim != 2:
        return np.nan

    K_avail = A.shape[0]
    k = min(k, K_avail)

    total = np.sum(A ** 2)
    if total == 0:
        return np.nan

    part = np.sum(A[:k, :] ** 2)
    return part / total


def collect_runs():
    rows = []

    for sub_dir in sorted(PANG.glob("sub-*")):
        if not sub_dir.is_dir():
            continue

        sub = sub_dir.name

        for run_dir in sorted(sub_dir.glob("run-*")):
            if not run_dir.is_dir():
                continue

            try:
                run = int(run_dir.name.split("-")[1])
            except Exception:
                continue

            for hemi in ["L", "R"]:
                a_path = run_dir / f"A_{hemi}.npy"
                if not a_path.exists():
                    continue

                A = np.load(a_path)

                for k in K_VALUES:
                    r2 = cumulative_energy_fraction(A, k)
                    rows.append({
                        "subject": sub,
                        "run": run,
                        "hemi": hemi,
                        "K": k,
                        "energy_fraction": r2,
                    })

    if not rows:
        raise RuntimeError("No A_L.npy / A_R.npy files found in pang_out/sub-*/run-*")

    return pd.DataFrame(rows)


def main():
    df = collect_runs()

    # Save all run-level values
    out_all = TABLE_DIR / "table_cumulative_eigenmode_energy_allruns.csv"
    df.to_csv(out_all, index=False)

    # Subject-level mean across runs and hemispheres
    subj = (
        df.groupby(["subject", "K"], as_index=False)
        .agg(energy_subject=("energy_fraction", "mean"))
    )

    out_subj = TABLE_DIR / "table_cumulative_eigenmode_energy_subject_level.csv"
    subj.to_csv(out_subj, index=False)

    # Group mean and SEM across subjects
    group = (
        subj.groupby("K", as_index=False)
        .agg(
            energy_mean=("energy_subject", "mean"),
            energy_sem=("energy_subject", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            N_subj=("subject", "nunique"),
        )
    )

    out_group = TABLE_DIR / "table_cumulative_eigenmode_energy_group.csv"
    group.to_csv(out_group, index=False)

    # ----------------------------
    # Plot
    # ----------------------------
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    K = group["K"].values
    mean = group["energy_mean"].values
    sem = group["energy_sem"].values

    # Individual subject curves
    for _, g in subj.groupby("subject"):
        g = g.sort_values("K")
        ax.plot(
            g["K"].values,
            g["energy_subject"].values,
            linewidth=0.8,
            alpha=0.18,
        )

    # Group mean
    ax.plot(
        K,
        mean,
        "-o",
        lw=2.5,
        markersize=5,
        label="Group mean",
    )

    # SEM band
    ax.fill_between(
        K,
        mean - sem,
        mean + sem,
        alpha=0.22,
        label="SEM",
    )

    # Key vertical markers
    key_K = [1, 3, 10, 20]
    for k in key_K:
        ax.axvline(k, linestyle="--", linewidth=1, alpha=0.45)

    # Annotate key points
    for k in key_K:
        idx = np.where(K == k)[0]
        if len(idx) == 0:
            continue
        i = idx[0]
        ax.text(
            K[i],
            mean[i] + 0.015,
            f"{mean[i] * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Number of eigenmodes (K)")
    ax.set_ylabel("Cumulative eigenmode energy fraction")
    ax.set_title("Low-dimensional structure of cortical dynamics")

    ax.set_ylim(0.75, 1.02)

    # Log x-axis helps show early saturation
    ax.set_xscale("log")
    ax.set_xticks(K)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.grid(False)
    ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()

    out_png = OUT_DIR / "figure_cumulative_eigenmode_energy.png"
    out_pdf = OUT_DIR / "figure_cumulative_eigenmode_energy.pdf"

    plt.savefig(out_png, dpi=300, facecolor="white")
    plt.savefig(out_pdf, facecolor="white")
    plt.close()

    print(f"✅ wrote {out_all}")
    print(f"✅ wrote {out_subj}")
    print(f"✅ wrote {out_group}")
    print(f"✅ wrote {out_png}")
    print(f"✅ wrote {out_pdf}")

    print("\nGroup summary:")
    print(group)


if __name__ == "__main__":
    main()