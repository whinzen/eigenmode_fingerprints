from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"
OUT_DIR = BASE / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES_L = {
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


def load_bihemi(path_l: Path, path_r: Path) -> pd.DataFrame:
    dl = pd.read_csv(path_l)
    dr = pd.read_csv(path_r)

    for df in (dl, dr):
        if "k" in df.columns and "mode_k" not in df.columns:
            df.rename(columns={"k": "mode_k"}, inplace=True)

    dl = dl.sort_values("mode_k").reset_index(drop=True)
    dr = dr.sort_values("mode_k").reset_index(drop=True)

    merged = dl.merge(dr, on="mode_k", suffixes=("_L", "_R"), how="inner")
    merged = merged[merged["mode_k"] > 0].copy()

    merged["beta_mean"] = (merged["beta_mean_L"] + merged["beta_mean_R"]) / 2.0
    merged["beta_sem"] = np.sqrt(
        merged["beta_sem_L"].fillna(0.0) ** 2 +
        merged["beta_sem_R"].fillna(0.0) ** 2
    ) / 2.0

    return merged[["mode_k", "beta_mean", "beta_sem"]].sort_values("mode_k")


def main():
    metrics = {}
    for name in FILES_L:
        metrics[name] = load_bihemi(FILES_L[name], FILES_R[name])

    shared_modes = set(metrics["Shift"]["mode_k"])
    for df in metrics.values():
        shared_modes &= set(df["mode_k"])
    shared_modes = sorted(shared_modes)

    for name in metrics:
        metrics[name] = metrics[name][metrics[name]["mode_k"].isin(shared_modes)].copy()

    shift = metrics["Shift"][["mode_k", "beta_mean"]].rename(columns={"beta_mean": "beta_shift"})

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for name in ["Prediction error (AR)", "Subspace exit", "Curvature"]:
        df = metrics[name].merge(shift, on="mode_k", how="inner")
        df["delta_from_shift"] = df["beta_mean"] - df["beta_shift"]

        ax.plot(
            df["mode_k"],
            df["delta_from_shift"],
            linewidth=2,
            label=name
        )

    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("Eigenmode index (k)")
    ax.set_ylabel("Δβ relative to shift")
    ax.set_title("Profiles relative to shift")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    png = OUT_DIR / "profiles_minus_shift.png"
    pdf = OUT_DIR / "profiles_minus_shift.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ wrote {png}")
    print(f"✅ wrote {pdf}")

    print("\n=== max |Δβ| relative to shift ===")
    for name in ["Prediction error (AR)", "Subspace exit", "Curvature"]:
        df = metrics[name].merge(shift, on="mode_k", how="inner")
        max_abs = np.max(np.abs(df["beta_mean"] - df["beta_shift"]))
        print(f"{name:25s} {max_abs:.6f}")


if __name__ == "__main__":
    main()