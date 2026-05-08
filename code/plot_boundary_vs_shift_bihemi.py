from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from settings import PANG_OUT

BOUND_DIR = PANG_OUT / "group_boundary_glm"
SHIFT_DIR = PANG_OUT / "group_sentence_shift_glm"
OUT_DIR = PANG_OUT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_boundary(hemi):
    f = BOUND_DIR / f"group_onset_hemi-{hemi}_subjectlevel_by_mode_excl_k0.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing boundary file: {f}")
    return pd.read_csv(f)

def load_shift(hemi):
    f = SHIFT_DIR / f"group_sentence_shift_hemi-{hemi}_by_mode_subject_level.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing sentence-shift file: {f}")
    return pd.read_csv(f)

def bihemi_average(dfL, dfR):
    cols_needed = ["mode_k", "lam", "beta_mean", "beta_sem"]
    dfL = dfL[cols_needed].copy()
    dfR = dfR[cols_needed].copy()

    merged = dfL.merge(dfR, on="mode_k", suffixes=("_L", "_R"))

    out = pd.DataFrame({
        "mode_k": merged["mode_k"],
        "lam": (merged["lam_L"] + merged["lam_R"]) / 2.0,
        "beta_mean": (merged["beta_mean_L"] + merged["beta_mean_R"]) / 2.0,
        "beta_sem": np.sqrt(merged["beta_sem_L"]**2 + merged["beta_sem_R"]**2) / 2.0,
    })
    return out.sort_values("mode_k")

def plot_vs_mode(bound_bi, shift_bi):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(bound_bi["mode_k"], bound_bi["beta_mean"], linewidth=2, label="Sentence boundary")
    ax.fill_between(
        bound_bi["mode_k"],
        bound_bi["beta_mean"] - bound_bi["beta_sem"],
        bound_bi["beta_mean"] + bound_bi["beta_sem"],
        alpha=0.2
    )

    ax.plot(shift_bi["mode_k"], shift_bi["beta_mean"], linewidth=2, label="Sentence shift")
    ax.fill_between(
        shift_bi["mode_k"],
        shift_bi["beta_mean"] - shift_bi["beta_sem"],
        shift_bi["beta_mean"] + shift_bi["beta_sem"],
        alpha=0.2
    )

    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_xlabel("Mode index k")
    ax.set_ylabel("β")
    ax.set_title("Boundary vs sentence shift (bihemispheric average)")
    ax.legend(frameon=False)
    fig.tight_layout()

    out = OUT_DIR / "boundary_vs_shift_bihemi_by_mode.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"✅ wrote {out}")

def plot_vs_lambda(bound_bi, shift_bi):
    bound_bi = bound_bi.sort_values("lam")
    shift_bi = shift_bi.sort_values("lam")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(bound_bi["lam"], bound_bi["beta_mean"], linewidth=2, label="Sentence boundary")
    ax.fill_between(
        bound_bi["lam"],
        bound_bi["beta_mean"] - bound_bi["beta_sem"],
        bound_bi["beta_mean"] + bound_bi["beta_sem"],
        alpha=0.2
    )

    ax.plot(shift_bi["lam"], shift_bi["beta_mean"], linewidth=2, label="Sentence shift")
    ax.fill_between(
        shift_bi["lam"],
        shift_bi["beta_mean"] - shift_bi["beta_sem"],
        shift_bi["beta_mean"] + shift_bi["beta_sem"],
        alpha=0.2
    )

    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Eigenvalue λ")
    ax.set_ylabel("β")
    ax.set_title("Boundary vs sentence shift (bihemispheric average)")
    ax.legend(frameon=False)
    fig.tight_layout()

    out = OUT_DIR / "boundary_vs_shift_bihemi_by_lambda.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"✅ wrote {out}")

def main():
    bound_L = load_boundary("L")
    bound_R = load_boundary("R")
    shift_L = load_shift("L")
    shift_R = load_shift("R")

    bound_bi = bihemi_average(bound_L, bound_R)
    shift_bi = bihemi_average(shift_L, shift_R)

    bound_bi.to_csv(OUT_DIR / "boundary_bihemi_by_mode.csv", index=False)
    shift_bi.to_csv(OUT_DIR / "sentence_shift_bihemi_by_mode.csv", index=False)

    plot_vs_mode(bound_bi, shift_bi)
    plot_vs_lambda(bound_bi, shift_bi)

if __name__ == "__main__":
    main()