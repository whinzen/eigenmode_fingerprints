from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"
OUT_DIR = BASE / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    ("shift", "Shift"),
    ("pred_error_ar", "AR prediction error"),
    ("pred_error_subspace", "Subspace-exit error"),
    ("curvature", "Curvature"),
]

def load_metric(metric):
    left = BASE / f"group_{metric}_glm" / f"group_{metric}_hemi-L_by_mode_subject_level.csv"
    right = BASE / f"group_{metric}_glm" / f"group_{metric}_hemi-R_by_mode_subject_level.csv"

    dl = pd.read_csv(left)
    dr = pd.read_csv(right)

    merged = dl.merge(dr, on="mode_k", suffixes=("_L", "_R"))
    merged["beta_mean"] = (merged["beta_mean_L"] + merged["beta_mean_R"]) / 2.0
    merged["beta_sem"] = np.sqrt(
        (merged["beta_sem_L"].fillna(0) ** 2 + merged["beta_sem_R"].fillna(0) ** 2)
    ) / 2.0

    return merged[["mode_k", "beta_mean", "beta_sem"]].sort_values("mode_k")

def normalize_curve(y):
    y = np.asarray(y, float)
    m = np.nanmax(np.abs(y))
    if not np.isfinite(m) or m == 0:
        return y
    return y / m

def plot_metrics(metric_dfs):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    offsets = np.linspace(0.0, 0.3, len(metric_dfs))  # small vertical offsets

    for (df, label), off in zip(metric_dfs, offsets):
        y = normalize_curve(df["beta_mean"].values)

        ax.plot(
            df["mode_k"],
            y + off,
            linewidth=2,
            label=label
        )

    ax.set_xlabel("Eigenmode index (k)")
    ax.set_ylabel("Normalized beta (offset for visibility)")
    ax.set_title("Fine-grained transition metrics (normalized profiles)")

    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure2_new_transition_metrics.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure2_new_transition_metrics.pdf", bbox_inches="tight")

    print("✅ wrote Figure 2 (normalized + offset)")

def main():
    metric_dfs = []
    for metric, label in METRICS:
        df = load_metric(metric)
        metric_dfs.append((df, label))
    plot_metrics(metric_dfs)

if __name__ == "__main__":
    main()