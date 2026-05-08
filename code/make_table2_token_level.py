from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out" / "word_transition_geometry"
OUT_DIR = Path.home() / "eigenmode_fingerprints" / "pang_out" / "paper_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = BASE / "word_transition_summary.csv"
RAW_PATH = BASE / "word_transition_geometry.csv"

OUT_CSV_A = OUT_DIR / "table2A_token_level_effects.csv"
OUT_CSV_B = OUT_DIR / "table2B_token_level_correlations.csv"


def p_to_str(p):
    if pd.isna(p):
        return ""
    if p == 0:
        return "<1e-300"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def cohens_d_from_summary(m1, m2, sem1, sem2, n1, n2):
    """
    Reconstruct pooled SD approximately from SEMs and sample sizes.
    """
    if n1 < 2 or n2 < 2:
        return np.nan

    sd1 = sem1 * np.sqrt(n1)
    sd2 = sem2 * np.sqrt(n2)

    pooled = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan

    return (m2 - m1) / pooled


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_PATH}")
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw file: {RAW_PATH}")

    summary = pd.read_csv(SUMMARY_PATH)
    raw = pd.read_csv(RAW_PATH)

    metrics = [
        "shift",
        "pred_error_ar",
        "pred_error_subspace",
        "curvature",
    ]

    # -------------------------
    # Table 2A: boundary effects
    # -------------------------
    rows = []
    for metric in metrics:
        row = summary.loc[summary["measure"] == metric]
        if row.empty:
            raise RuntimeError(f"Metric not found in summary file: {metric}")
        row = row.iloc[0]

        delta = row["boundary_mean"] - row["within_mean"]
        d = cohens_d_from_summary(
            row["within_mean"], row["boundary_mean"],
            row["within_sem"], row["boundary_sem"],
            int(row["n_within"]), int(row["n_boundary"])
        )

        pretty_name = {
            "shift": "Shift",
            "pred_error_ar": "Prediction error (AR)",
            "pred_error_subspace": "Subspace exit",
            "curvature": "Curvature",
        }[metric]

        rows.append({
            "Metric": pretty_name,
            "Within mean": round(row["within_mean"], 2),
            "Boundary mean": round(row["boundary_mean"], 2),
            "Δ (Boundary − Within)": round(delta, 2),
            "Cohen's d": round(d, 2),
            "t": round(row["t"], 2),
            "p": p_to_str(row["p"]),
            "n_within": int(row["n_within"]),
            "n_boundary": int(row["n_boundary"]),
        })

    tableA = pd.DataFrame(rows)
    tableA.to_csv(OUT_CSV_A, index=False)

    # -------------------------
    # Table 2B: inter-metric correlations
    # -------------------------
    corr_df = raw[metrics].copy()
    corr = corr_df.corr().round(2)

    corr = corr.rename(index={
        "shift": "Shift",
        "pred_error_ar": "Prediction error (AR)",
        "pred_error_subspace": "Subspace exit",
        "curvature": "Curvature",
    }, columns={
        "shift": "Shift",
        "pred_error_ar": "Prediction error (AR)",
        "pred_error_subspace": "Subspace exit",
        "curvature": "Curvature",
    })

    corr.to_csv(OUT_CSV_B)

    print(f"✅ wrote {OUT_CSV_A}")
    print(f"✅ wrote {OUT_CSV_B}\n")

    print("=== Table 2A: boundary effects ===")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(tableA)

    print("\n=== Table 2B: correlations ===")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(corr)


if __name__ == "__main__":
    main()