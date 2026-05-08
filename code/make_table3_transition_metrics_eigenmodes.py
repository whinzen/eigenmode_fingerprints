from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"
OUT_DIR = BASE / "paper_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "Shift": BASE / "group_shift_glm" / "group_shift_hemi-L_by_mode_subject_level.csv",
    "Prediction error (AR)": BASE / "group_pred_error_ar_glm" / "group_pred_error_ar_hemi-L_by_mode_subject_level.csv",
    "Subspace exit": BASE / "group_pred_error_subspace_glm" / "group_pred_error_subspace_hemi-L_by_mode_subject_level.csv",
    "Curvature": BASE / "group_curvature_glm" / "group_curvature_hemi-L_by_mode_subject_level.csv",
}

FILES_R = {
    "Shift": BASE / "group_shift_glm" / "group_shift_hemi-R_by_mode_subject_level.csv",
    "Prediction error (AR)": BASE / "group_pred_error_ar_glm" / "group_pred_error_ar_hemi-R_by_mode_subject_level.csv",
    "Subspace exit": BASE / "group_pred_error_subspace_glm" / "group_pred_error_subspace_hemi-R_by_mode_subject_level.csv",
    "Curvature": BASE / "group_curvature_glm" / "group_curvature_hemi-R_by_mode_subject_level.csv",
}

OUT_CSV = OUT_DIR / "table3_transition_metrics_eigenmodes.csv"


def load_metric(left_path: Path, right_path: Path) -> pd.DataFrame:
    dl = pd.read_csv(left_path)
    dr = pd.read_csv(right_path)

    # standardize possible legacy names
    for df in (dl, dr):
        if "k" in df.columns and "mode_k" not in df.columns:
            df.rename(columns={"k": "mode_k"}, inplace=True)
        if "lambda" in df.columns and "lam" not in df.columns:
            df.rename(columns={"lambda": "lam"}, inplace=True)

    dl = dl.sort_values("mode_k").reset_index(drop=True)
    dr = dr.sort_values("mode_k").reset_index(drop=True)

    merged = dl.merge(dr, on="mode_k", suffixes=("_L", "_R"), how="inner")

    # exclude global mode
    merged = merged[merged["mode_k"] > 0].copy()

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


def summarize_profile(name: str, df: pd.DataFrame) -> dict:
    low = df[df["mode_k"] <= 10].copy()

    idx_peak = df["beta_mean"].abs().idxmax()
    peak_mode = int(df.loc[idx_peak, "mode_k"])
    peak_beta = float(df.loc[idx_peak, "beta_mean"])

    n_sig = int((df["sig_q05"] == 1).sum())
    max_sig_mode = int(df.loc[df["sig_q05"] == 1, "mode_k"].max()) if (df["sig_q05"] == 1).any() else np.nan

    return {
        "Metric": name,
        "Mean β (modes 1–10)": low["beta_mean"].mean(),
        "Peak mode k": peak_mode,
        "Peak β": peak_beta,
        "# significant modes (q<0.05)": n_sig,
        "Highest significant mode": max_sig_mode,
    }


def main():
    metrics = {}
    for name in FILES:
        metrics[name] = load_metric(FILES[name], FILES_R[name])

    # find shared modes across all metrics
    shared_modes = set(metrics["Shift"]["mode_k"])
    for name, df in metrics.items():
        shared_modes &= set(df["mode_k"])

    shared_modes = sorted(shared_modes)

    for name in metrics:
        metrics[name] = metrics[name][metrics[name]["mode_k"].isin(shared_modes)].copy()

    # correlations to shift profile
    shift_profile = metrics["Shift"]["beta_mean"].values
    profile_corrs = {}
    for name, df in metrics.items():
        r = np.corrcoef(shift_profile, df["beta_mean"].values)[0, 1]
        profile_corrs[name] = r

    rows = []
    for name, df in metrics.items():
        row = summarize_profile(name, df)
        row["Profile correlation with shift"] = profile_corrs[name]
        rows.append(row)

    table = pd.DataFrame(rows)

    # rounding
    table["Mean β (modes 1–10)"] = table["Mean β (modes 1–10)"].round(2)
    table["Peak β"] = table["Peak β"].round(2)
    table["Profile correlation with shift"] = table["Profile correlation with shift"].round(2)

    table.to_csv(OUT_CSV, index=False)

    print(f"✅ wrote {OUT_CSV}\n")
    print(f"Shared modes used: {len(shared_modes)} ({min(shared_modes)}..{max(shared_modes)})\n")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(table)


if __name__ == "__main__":
    main()