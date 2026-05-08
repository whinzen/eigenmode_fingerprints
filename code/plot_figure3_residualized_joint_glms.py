from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"
OUT_DIR = BASE / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("shift", "pred_error_ar"),
    ("shift", "pred_error_subspace"),
    ("shift", "curvature"),
]

def normalize_curve(y):
    y = np.asarray(y, float)
    m = np.nanmax(np.abs(y))
    if not np.isfinite(m) or m == 0:
        return y
    return y / m

def load_pair_curve(pair_name, metric_name):
    left = BASE / f"group_{pair_name}_glm" / f"group_{pair_name}_{metric_name}_hemi-L_by_mode_subject_level.csv"
    right = BASE / f"group_{pair_name}_glm" / f"group_{pair_name}_{metric_name}_hemi-R_by_mode_subject_level.csv"

    dl = pd.read_csv(left)
    dr = pd.read_csv(right)

    merged = dl.merge(dr, on="mode_k", suffixes=("_L", "_R"))
    merged["beta_mean"] = (merged["beta_mean_L"] + merged["beta_mean_R"]) / 2.0
    return merged[["mode_k", "beta_mean"]].sort_values("mode_k")

def plot_residualized():
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True)

    for ax, (base_metric, extra_metric) in zip(axes, PAIRS):
        pair_name = f"{base_metric}__plus__{extra_metric}_resid"

        d_base = load_pair_curve(pair_name, base_metric)
        d_extra = load_pair_curve(pair_name, f"{extra_metric}_resid")

        y1 = normalize_curve(d_base["beta_mean"].values)
        y2 = normalize_curve(d_extra["beta_mean"].values)

        good = np.isfinite(y1) & np.isfinite(y2)
        r = np.corrcoef(y1[good], y2[good])[0, 1] if good.sum() > 2 else np.nan

        ax.plot(d_base["mode_k"], y1, color="black", linewidth=2, label="Shift")
        ax.plot(d_extra["mode_k"], y2, color="darkorange", linewidth=2, linestyle="--",
                label=f"{extra_metric}_resid")

        ax.axhline(0, color="gray", linewidth=1)
        ax.set_title(f"{extra_metric}\nr = {r:.3f}")
        ax.set_xlabel("Eigenmode index (k)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_ylabel("Normalized beta profile")
    fig.suptitle("Residualized joint GLMs preserve the same mode-profile shape", y=1.02)
    fig.tight_layout()

    png = OUT_DIR / "figure3_residualized_joint_glms.png"
    pdf = OUT_DIR / "figure3_residualized_joint_glms.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ wrote {png}")
    print(f"✅ wrote {pdf}")

if __name__ == "__main__":
    plot_residualized()