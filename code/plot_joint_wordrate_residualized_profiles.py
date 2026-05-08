from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"
IN_DIR = BASE / "group_joint_wordrate_resid"
OUT_DIR = BASE / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = {
    "shift": "Shift",
    "pred_error_ar": "Prediction error (AR)",
    "pred_error_subspace": "Subspace exit",
    "curvature": "Curvature",
}


def load_bihemi(metric_key: str) -> pd.DataFrame:
    f = IN_DIR / f"group_joint_wordrate_resid_{metric_key}_by_mode_subject_level.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing file: {f}")

    df = pd.read_csv(f).copy()
    dl = df[df["hemi"] == "L"].copy()
    dr = df[df["hemi"] == "R"].copy()

    merged = dl.merge(dr, on=["metric", "mode_k"], suffixes=("_L", "_R"), how="inner")

    merged["beta_wordrate_mean"] = (
        merged["beta_wordrate_mean_L"] + merged["beta_wordrate_mean_R"]
    ) / 2.0
    merged["beta_wordrate_sem"] = np.sqrt(
        merged["beta_wordrate_sem_L"].fillna(0.0) ** 2 +
        merged["beta_wordrate_sem_R"].fillna(0.0) ** 2
    ) / 2.0

    merged["beta_metric_resid_mean"] = (
        merged["beta_metric_resid_mean_L"] + merged["beta_metric_resid_mean_R"]
    ) / 2.0
    merged["beta_metric_resid_sem"] = np.sqrt(
        merged["beta_metric_resid_sem_L"].fillna(0.0) ** 2 +
        merged["beta_metric_resid_sem_R"].fillna(0.0) ** 2
    ) / 2.0

    return merged.sort_values("mode_k")


def summarize(df: pd.DataFrame):
    d = df[df["mode_k"] > 0].copy()
    low = d[d["mode_k"].between(1, 10)].copy()

    wr_peak_idx = d["beta_wordrate_mean"].abs().idxmax()
    met_peak_idx = d["beta_metric_resid_mean"].abs().idxmax()

    return {
        "wr_low_mean": float(low["beta_wordrate_mean"].mean()),
        "met_low_mean": float(low["beta_metric_resid_mean"].mean()),
        "wr_peak_mode": int(d.loc[wr_peak_idx, "mode_k"]),
        "met_peak_mode": int(d.loc[met_peak_idx, "mode_k"]),
        "wr_peak_beta": float(d.loc[wr_peak_idx, "beta_wordrate_mean"]),
        "met_peak_beta": float(d.loc[met_peak_idx, "beta_metric_resid_mean"]),
    }


def main():
    plt.style.use("default")

    # ---------------------------
    # Main panel figure
    # ---------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), sharex=True, sharey=False)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()

    for ax, (metric_key, metric_label) in zip(axes, METRICS.items()):
        df = load_bihemi(metric_key)
        df = df[df["mode_k"] > 0].copy()
        s = summarize(df)

        ax.set_facecolor("white")

        ax.plot(
            df["mode_k"], df["beta_wordrate_mean"],
            linewidth=2, label="Word rate"
        )
        ax.plot(
            df["mode_k"], df["beta_metric_resid_mean"],
            linewidth=2, label=f"{metric_label} (residualized)"
        )
        ax.axhline(0, color="gray", linewidth=1)

        ax.set_title(metric_label)
        ax.set_xlabel("Eigenmode index (k)")
        ax.set_ylabel("Mean β")
        ax.grid(False)

        txt = (
            f"Word rate mean β (1–10): {s['wr_low_mean']:.2f}\n"
            f"Residualized mean β (1–10): {s['met_low_mean']:.2f}\n"
            f"Peak k: {s['wr_peak_mode']} / {s['met_peak_mode']}\n"
            f"Peak β: {s['wr_peak_beta']:.2f} / {s['met_peak_beta']:.2f}"
        )
       
        ax.text(
    		0.97, 0.03, txt,
    		transform=ax.transAxes,
    		fontsize=8.5,
    		va="bottom",
    		ha="right",
    		bbox=dict(facecolor="white", alpha=0.9, edgecolor="lightgray")
		)

    axes[0].legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "joint_wordrate_residualized_profiles.png", dpi=300, facecolor="white")
    plt.savefig(OUT_DIR / "joint_wordrate_residualized_profiles.pdf", dpi=300, facecolor="white")
    plt.close()

    # ---------------------------
    # Scatter figure: same axis, opposite sign
    # ---------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()

    for ax, (metric_key, metric_label) in zip(axes, METRICS.items()):
        df = load_bihemi(metric_key)
        df = df[df["mode_k"] > 0].copy()

        x = df["beta_wordrate_mean"].values
        y = df["beta_metric_resid_mean"].values

        ax.set_facecolor("white")
        ax.scatter(x, y, s=20)
        ax.axhline(0, color="gray", linewidth=1)
        ax.axvline(0, color="gray", linewidth=1)

        # least-squares line for display
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, m * xx + b, linewidth=1.5)

        ax.set_title(metric_label)
        ax.set_xlabel("Word rate β")
        ax.set_ylabel("Residualized metric β")
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "joint_wordrate_residualized_scatter.png", dpi=300, facecolor="white")
    plt.savefig(OUT_DIR / "joint_wordrate_residualized_scatter.pdf", dpi=300, facecolor="white")
    plt.close()

    print(f"✅ wrote {OUT_DIR / 'joint_wordrate_residualized_profiles.png'}")
    print(f"✅ wrote {OUT_DIR / 'joint_wordrate_residualized_profiles.pdf'}")
    print(f"✅ wrote {OUT_DIR / 'joint_wordrate_residualized_scatter.png'}")
    print(f"✅ wrote {OUT_DIR / 'joint_wordrate_residualized_scatter.pdf'}")


if __name__ == "__main__":
    main()