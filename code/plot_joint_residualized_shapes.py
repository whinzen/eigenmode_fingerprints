import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path("~/eigenmode_fingerprints/pang_out").expanduser()
OUT_DIR = BASE / "group_joint_residualized_shape_plots_fixed"
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

def load_curve(pair_name, metric_name, hemi):
    pair_dir = BASE / f"group_{pair_name}_glm"
    f = pair_dir / f"group_{pair_name}_{metric_name}_hemi-{hemi}_by_mode_subject_level.csv"
    if not f.exists():
        return None
    return pd.read_csv(f)

def make_plot(hemi, normalized=True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, (base_metric, extra_metric) in zip(axes, PAIRS):
        pair_name = f"{base_metric}__plus__{extra_metric}_resid"

        d_base = load_curve(pair_name, base_metric, hemi)
        d_extra = load_curve(pair_name, f"{extra_metric}_resid", hemi)

        if d_base is None or d_extra is None:
            ax.set_title(f"{pair_name}\nmissing")
            ax.axis("off")
            continue

        x1 = d_base["mode_k"].values
        y1 = d_base["beta_mean"].values.astype(float)

        x2 = d_extra["mode_k"].values
        y2 = d_extra["beta_mean"].values.astype(float)

        if normalized:
            y1_plot = normalize_curve(y1)
            y2_plot = normalize_curve(y2)
            ylabel = "Normalized beta profile"
            suffix = "normalized"
        else:
            y1_plot = y1
            y2_plot = y2
            ylabel = "Beta"
            suffix = "raw"

        # correlation in plotted space
        good = np.isfinite(y1_plot) & np.isfinite(y2_plot)
        r = np.corrcoef(y1_plot[good], y2_plot[good])[0, 1] if good.sum() > 2 else np.nan

        # base metric: black solid
        ax.plot(x1, y1_plot, color="black", linewidth=2, label=base_metric, zorder=3)

        # residualized metric: orange dashed
        ax.plot(
            x2,
            y2_plot,
            color="darkorange",
            linewidth=2,
            linestyle="--",
            label=f"{extra_metric}_resid",
            zorder=4
        )

        ax.axhline(0, color="gray", linewidth=1)
        ax.set_title(f"{pair_name}\nr = {r:.3f}")
        ax.set_xlabel("Mode k")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"Residualized joint GLM profiles ({suffix}, hemi {hemi})", y=1.02)
    fig.tight_layout()

    out_png = OUT_DIR / f"joint_residualized_shapes_{suffix}_hemi-{hemi}.png"
    out_pdf = OUT_DIR / f"joint_residualized_shapes_{suffix}_hemi-{hemi}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ wrote {out_png}")
    print(f"✅ wrote {out_pdf}")

for hemi in ["L", "R"]:
    make_plot(hemi, normalized=False)
    make_plot(hemi, normalized=True)