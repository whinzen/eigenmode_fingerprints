from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"
IN_DIR = BASE / "group_joint_wordrate"
OUT_DIR = BASE / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = {
    "shift": "Shift",
    "pred_error_ar": "Prediction error (AR)",
    "pred_error_subspace": "Subspace exit",
    "curvature": "Curvature",
}


def load_bihemi(metric_key: str) -> pd.DataFrame:
    f = IN_DIR / f"group_joint_wordrate_{metric_key}_by_mode_subject_level.csv"
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

    merged["beta_metric_mean"] = (
        merged["beta_metric_mean_L"] + merged["beta_metric_mean_R"]
    ) / 2.0
    merged["beta_metric_sem"] = np.sqrt(
        merged["beta_metric_sem_L"].fillna(0.0) ** 2 +
        merged["beta_metric_sem_R"].fillna(0.0) ** 2
    ) / 2.0

    return merged[
        [
            "metric",
            "mode_k",
            "beta_wordrate_mean",
            "beta_wordrate_sem",
            "beta_metric_mean",
            "beta_metric_sem",
        ]
    ].sort_values("mode_k")


def zscore(v):
    v = np.asarray(v, float)
    s = v.std(ddof=0)
    if s == 0 or not np.isfinite(s):
        return v - v.mean()
    return (v - v.mean()) / s


def summarize_profile(name: str, df: pd.DataFrame):
    low = df[df["mode_k"].between(1, 10)].copy()
    wr = df[df["mode_k"] > 0]["beta_wordrate_mean"].values
    met = df[df["mode_k"] > 0]["beta_metric_mean"].values
    r = np.corrcoef(wr, met)[0, 1]

    out = {
        "Metric": name,
        "Word rate mean β (modes 1–10)": low["beta_wordrate_mean"].mean(),
        "Metric mean β (modes 1–10)": low["beta_metric_mean"].mean(),
        "Word rate peak mode k": int(df.loc[df["beta_wordrate_mean"].abs().idxmax(), "mode_k"]),
        "Metric peak mode k": int(df.loc[df["beta_metric_mean"].abs().idxmax(), "mode_k"]),
        "Word rate peak β": float(df["beta_wordrate_mean"].iloc[df["beta_wordrate_mean"].abs().idxmax()]),
        "Metric peak β": float(df["beta_metric_mean"].iloc[df["beta_metric_mean"].abs().idxmax()]),
        "Profile correlation (word rate vs metric)": r,
    }
    return out


def main():
    plt.style.use("default")

    # ---------- Panel figure: raw profiles ----------
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=False)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()

    summary_rows = []

    for ax, (metric_key, metric_label) in zip(axes, METRICS.items()):
        df = load_bihemi(metric_key)
        df = df[df["mode_k"] > 0].copy()

        ax.set_facecolor("white")
        ax.plot(df["mode_k"], df["beta_wordrate_mean"], linewidth=2, label="Word rate")
        ax.plot(df["mode_k"], df["beta_metric_mean"], linewidth=2, label=metric_label)
        ax.axhline(0, color="gray", linewidth=1)

        ax.set_title(metric_label)
        ax.set_xlabel("Eigenmode index (k)")
        ax.set_ylabel("Mean β")
        ax.grid(False)

        summary_rows.append(summarize_profile(metric_label, df))

    axes[0].legend(frameon=False, fontsize=9)
    plt.tight_layout()
    out_png = OUT_DIR / "joint_wordrate_controls_raw.png"
    out_pdf = OUT_DIR / "joint_wordrate_controls_raw.pdf"
    plt.savefig(out_png, dpi=300, facecolor="white")
    plt.savefig(out_pdf, dpi=300, facecolor="white")
    plt.close()

    # ---------- Panel figure: normalized profiles ----------
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()

    for ax, (metric_key, metric_label) in zip(axes, METRICS.items()):
        df = load_bihemi(metric_key)
        df = df[df["mode_k"] > 0].copy()

        ax.set_facecolor("white")
        ax.plot(df["mode_k"], zscore(df["beta_wordrate_mean"].values), linewidth=2, label="Word rate")
        ax.plot(df["mode_k"], zscore(df["beta_metric_mean"].values), linewidth=2, label=metric_label)
        ax.axhline(0, color="gray", linewidth=1)

        ax.set_title(metric_label)
        ax.set_xlabel("Eigenmode index (k)")
        ax.set_ylabel("Z-scored β")
        ax.grid(False)

    axes[0].legend(frameon=False, fontsize=9)
    plt.tight_layout()
    out_png = OUT_DIR / "joint_wordrate_controls_zscore.png"
    out_pdf = OUT_DIR / "joint_wordrate_controls_zscore.pdf"
    plt.savefig(out_png, dpi=300, facecolor="white")
    plt.savefig(out_pdf, dpi=300, facecolor="white")
    plt.close()

    # ---------- Summary table ----------
    summary = pd.DataFrame(summary_rows)
    summary["Word rate mean β (modes 1–10)"] = summary["Word rate mean β (modes 1–10)"].round(2)
    summary["Metric mean β (modes 1–10)"] = summary["Metric mean β (modes 1–10)"].round(2)
    summary["Word rate peak β"] = summary["Word rate peak β"].round(2)
    summary["Metric peak β"] = summary["Metric peak β"].round(2)
    summary["Profile correlation (word rate vs metric)"] = summary["Profile correlation (word rate vs metric)"].round(3)

    out_csv = BASE / "paper_tables" / "table_joint_wordrate_controls.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    print(f"✅ wrote {out_png}")
    print(f"✅ wrote {out_pdf}")
    print(f"✅ wrote {BASE / 'paper_figures' / 'joint_wordrate_controls_raw.png'}")
    print(f"✅ wrote {BASE / 'paper_figures' / 'joint_wordrate_controls_raw.pdf'}")
    print(f"✅ wrote {out_csv}\n")

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary)


if __name__ == "__main__":
    main()