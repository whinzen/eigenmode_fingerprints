#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

MEAN_GROUP = PANG / "subcortex" / "hipp_mean_signal_group"
ENERGY_GROUP = PANG / "subcortex" / "hippocampus_group"
COV_GROUP = PANG / "subcortex" / "hipp_trajectory_mean_covariate_group"

OUT = PANG / "subcortex" / "hipp_summary_figures"
OUT.mkdir(parents=True, exist_ok=True)

ORDER = [
    "sentence_boundary",
    "sentence_shift",
    "token_shift",
    "pred_error_ar",
    "pred_error_subspace",
    "curvature",
]

LABELS = {
    "sentence_boundary": "Sentence\nboundary",
    "sentence_shift": "Sentence\nshift",
    "token_shift": "Token\nshift",
    "pred_error_ar": "AR\nerror",
    "pred_error_subspace": "Subspace\nerror",
    "curvature": "Curvature",
}

ENERGY_MAP = {
    "sentence_boundary": "boundary",
    "sentence_shift": "sentence_shift",
    "token_shift": "token_shift",
    "pred_error_ar": "pred_error_ar",
    "pred_error_subspace": "pred_error_subspace",
    "curvature": "curvature",
}


def stars(p):
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_stars(ax, x, y, err, pvals):
    y_min, y_max = ax.get_ylim()
    offset = 0.04 * (y_max - y_min)

    for xi, yi, ei, p in zip(x, y, err, pvals):
        s = stars(p)
        ypos = yi - ei - offset if yi < 0 else yi + ei + offset
        va = "top" if yi < 0 else "bottom"
        ax.text(xi, ypos, s, ha="center", va=va, fontsize=10)


def load_mean():
    f = MEAN_GROUP / "group_all_predictors_hipp_mean_signal_bihemi.csv"
    df = pd.read_csv(f)
    df["predictor"] = pd.Categorical(df["predictor"], ORDER, ordered=True)
    return df.sort_values("predictor")


def load_energy():
    rows = []
    for pred, metric in ENERGY_MAP.items():
        f = ENERGY_GROUP / f"group_{metric}_hipp_bihemi_by_mode_subject_level.csv"
        if not f.exists():
            print(f"[skip] missing energy file: {f}")
            continue
        d = pd.read_csv(f)
        d["predictor"] = pred
        rows.append(d)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def load_covariate():
    f = COV_GROUP / "group_all_trajectory_mean_covariate_bihemi.csv"
    if not f.exists():
        print(f"[skip] missing covariate group file: {f}")
        return pd.DataFrame()
    return pd.read_csv(f)


def main():
    mean = load_mean()
    energy = load_energy()
    cov = load_covariate()

    fig = plt.figure(figsize=(14, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.4, 0.9])

    # Panel A
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(mean)) * 1.18
    y = mean["beta_mean"].values
    err = mean["beta_sem"].values
    pvals = mean["p"].values

    ax1.bar(x, y, yerr=err, capsize=4, width=0.58)
    ax1.axhline(0, linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS[str(p)] for p in mean["predictor"]])
    ax1.set_ylabel("β")
    ax1.set_title("A. Mean hippocampal signal")
    add_stars(ax1, x, y, err, pvals)

        # Panel B
    ax2 = fig.add_subplot(gs[0, 1])

    if not energy.empty:

        for pred in ORDER:

            d = energy[energy["predictor"] == pred].copy()

            if d.empty:
                continue

            d = d.sort_values("mode_k")

            x_mode = d["mode_k"].values
            y = d["beta_mean"].values

            if "beta_sem" in d.columns:
                sem = d["beta_sem"].values
            elif "beta_sem_bi" in d.columns:
                sem = d["beta_sem_bi"].values
            else:
                sem = None

            ax2.plot(
                x_mode,
                y,
                linewidth=1.8,
                label=LABELS[pred].replace("\n", " "),
            )

            if sem is not None:
                ax2.fill_between(
                    x_mode,
                    y - sem,
                    y + sem,
                    alpha=0.18,
                    linewidth=0,
                )

        ax2.axhline(0, linewidth=1)
        ax2.set_xlabel("Hippocampal eigenmode k")
        ax2.set_ylabel("β")
        ax2.set_title("B. Eigenmode energy")
        ax2.legend(
            frameon=False,
            fontsize=7,
            ncol=2,
            loc="best"
        )

    else:
        ax2.text(
            0.5,
            0.5,
            "Energy profiles not found",
            ha="center",
            va="center",
        )
        ax2.set_axis_off()

    # Panel C
    ax3 = fig.add_subplot(gs[0, 2])

    if not cov.empty:
        c = cov[cov["trajectory_column"] == "trajectory_step"].copy()
        if c.empty:
            c = cov.copy()

        effects = ["token_shift_control_mean", "mean_signal_covariate"]
        effect_labels = {
            "token_shift_control_mean": "Token shift\ncontrolling\nmean signal",
            "mean_signal_covariate": "Mean signal\ncovariate",
        }

        c = c[c["effect"].isin(effects)].copy()
        c["effect"] = pd.Categorical(c["effect"], effects, ordered=True)
        c = c.sort_values("effect")

        x = np.arange(len(c)) * 0.58
        y = c["beta_mean"].values
        err = c["beta_sem"].values
        pvals = c["p"].values

        ax3.bar(x, y, yerr=err, capsize=4, width=0.32)
        ax3.axhline(0, linewidth=1)
        ax3.set_xticks(x)
        ax3.set_xticklabels([effect_labels[str(e)] for e in c["effect"]])
        ax3.set_xlim(x.min() - 0.45, x.max() + 0.45)
        ax3.set_ylabel("β")
        ax3.set_title("C. Trajectory step GLM")
        add_stars(ax3, x, y, err, pvals)

    else:
        ax3.text(0.5, 0.5, "Trajectory results not found", ha="center", va="center")
        ax3.set_axis_off()

    fig.tight_layout()

    out_png = OUT / "hippocampus_all_results_summary.png"
    out_pdf = OUT / "hippocampus_all_results_summary.pdf"

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    caption = """Figure X. Hippocampal mean-signal, eigenmode-energy, and trajectory effects during naturalistic language comprehension. 
(A) Group-level GLM coefficients for mean hippocampal BOLD signal. Sentence-level predictors produced reliable negative effects, while token-level transition metrics showed substantially larger negative effects. 
(B) Hippocampal eigenmode-energy profiles across linguistic predictors. Unlike the cortical eigenmode results, hippocampal effects were broadly negative and showed little evidence for selective recruitment of specific spatial modes. 
(C) Trajectory step-length GLM including both token shift and mean hippocampal signal as predictors. Token shift remained strongly negative after controlling for mean signal, indicating that reduced hippocampal trajectory mobility is not reducible to global mean-signal suppression. Error bars indicate SEM across subjects. Stars denote one-sample tests against zero after subject-level run averaging: * p < .05, ** p < .01, *** p < .001, **** p < 1e-4.
"""

    caption_file = OUT / "hippocampus_all_results_summary_caption.txt"
    caption_file.write_text(caption)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {caption_file}")


if __name__ == "__main__":
    main()