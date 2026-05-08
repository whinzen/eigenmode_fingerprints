from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"
OUT_DIR = BASE / "paper_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_DIR = BASE / "group_sentence_level_glm"
BOUNDARY = GROUP_DIR / "group_boundary_by_mode_subject_level.csv"
SHIFT = GROUP_DIR / "group_sentence_shift_by_mode_subject_level.csv"

OUT_CSV = OUT_DIR / "table1_sentence_level.csv"


def load_metric(path, hemi):
    df = pd.read_csv(path)
    df = df[df["hemi"] == hemi].copy()
    return df[["mode_k", "beta_mean", "beta_sem", "p", "sig_q05"]].sort_values("mode_k")


def load_bihemi_shared(metric_path):
    dl = load_metric(metric_path, "L")
    dr = load_metric(metric_path, "R")

    merged = dl.merge(dr, on="mode_k", suffixes=("_L", "_R"), how="inner")

    merged["beta_mean"] = (merged["beta_mean_L"] + merged["beta_mean_R"]) / 2.0
    merged["beta_sem"] = np.sqrt(
        merged["beta_sem_L"].fillna(0.0) ** 2 +
        merged["beta_sem_R"].fillna(0.0) ** 2
    ) / 2.0
    merged["p"] = np.minimum(merged["p_L"], merged["p_R"])
    merged["sig_q05"] = (
        merged["sig_q05_L"].fillna(0).astype(int) |
        merged["sig_q05_R"].fillna(0).astype(int)
    )

    return merged[["mode_k", "beta_mean", "beta_sem", "p", "sig_q05"]].sort_values("mode_k")


def summarize_profile(name, df, low_mode_max=10):
    low = df[df["mode_k"] <= low_mode_max].copy()

    idx_peak = df["beta_mean"].abs().idxmax()
    peak_mode = int(df.loc[idx_peak, "mode_k"])
    peak_beta = float(df.loc[idx_peak, "beta_mean"])
    peak_abs_beta = abs(peak_beta)

    n_sig = int((df["sig_q05"] == 1).sum())
    max_sig_mode = int(df.loc[df["sig_q05"] == 1, "mode_k"].max()) if (df["sig_q05"] == 1).any() else np.nan

    return {
        "Regressor": name,
        "Mean β (modes 1–10)": low["beta_mean"].mean(),
        "Min β (modes 1–10)": low["beta_mean"].min(),
        "Max |β| (modes 1–10)": low["beta_mean"].abs().max(),
        "Peak mode k": peak_mode,
        "Peak β": peak_beta,
        "Peak |β|": peak_abs_beta,
        "# significant modes (q<0.05)": n_sig,
        "Highest significant mode": max_sig_mode,
        "p_min": float(df["p"].min()),
    }


def main():
    boundary = load_bihemi_shared(BOUNDARY)
    shift = load_bihemi_shared(SHIFT)

    # restrict both to shared modes only
    shared = boundary.merge(
        shift,
        on="mode_k",
        suffixes=("_boundary", "_shift"),
        how="inner"
    )
    shared = shared[shared["mode_k"] > 0].copy()

    boundary_shared = shared[[
        "mode_k", "beta_mean_boundary", "beta_sem_boundary", "p_boundary", "sig_q05_boundary"
    ]].rename(columns={
        "beta_mean_boundary": "beta_mean",
        "beta_sem_boundary": "beta_sem",
        "p_boundary": "p",
        "sig_q05_boundary": "sig_q05",
    })

    shift_shared = shared[[
        "mode_k", "beta_mean_shift", "beta_sem_shift", "p_shift", "sig_q05_shift"
    ]].rename(columns={
        "beta_mean_shift": "beta_mean",
        "beta_sem_shift": "beta_sem",
        "p_shift": "p",
        "sig_q05_shift": "sig_q05",
    })

    profile_r = np.corrcoef(
        boundary_shared["beta_mean"].values,
        shift_shared["beta_mean"].values
    )[0, 1]

    rows = [
        summarize_profile("Sentence boundary", boundary_shared),
        summarize_profile("Sentence shift", shift_shared),
    ]

    table = pd.DataFrame(rows)
    table["Profile correlation with sentence shift"] = [profile_r, profile_r]

    cols = [
        "Regressor",
        "Mean β (modes 1–10)",
        "Min β (modes 1–10)",
        "Max |β| (modes 1–10)",
        "Peak mode k",
        "Peak β",
        "Peak |β|",
        "# significant modes (q<0.05)",
        "Highest significant mode",
        "p_min",
        "Profile correlation with sentence shift",
    ]
    table = table[cols]

    # ✅ rounding (Nature-style)
    table_rounded = table.copy()

    round_2 = [
        "Mean β (modes 1–10)",
        "Min β (modes 1–10)",
        "Max |β| (modes 1–10)",
        "Peak β",
        "Peak |β|",
    ]

    for col in round_2:
        table_rounded[col] = table_rounded[col].round(2)

    table_rounded["Profile correlation with sentence shift"] = \
        table_rounded["Profile correlation with sentence shift"].round(2)

    # save rounded version
    table_rounded.to_csv(OUT_CSV, index=False)

    print(f"✅ wrote {OUT_CSV}\n")
    print("Using modes:", shared["mode_k"].min(), "to", shared["mode_k"].max(), "N =", len(shared))
    print(f"Shared modes used: {len(shared)} ({shared['mode_k'].min()}..{shared['mode_k'].max()})\n")

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(table_rounded)


if __name__ == "__main__":
    main()