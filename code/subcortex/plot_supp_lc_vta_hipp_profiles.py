#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

HC_GROUP = (
    PANG
    / "subcortex"
    / "hipp_mean_signal_group"
    / "group_all_predictors_hipp_mean_signal_bihemi.csv"
)

BS_GROUP = (
    PANG
    / "subcortex"
    / "brainstem_roi_group"
    / "group_brainstem_roi_glm_all_predictors.csv"
)

OUT = PANG / "subcortex" / "brainstem_vta_hipp_figures"
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

COLORS = {
    "HC": "#4C72B0",
    "VTA": "#C44E52",
    "LC_L": "#55A868",
    "LC_R": "#8172B2",
}


def load_profiles():
    hc = pd.read_csv(HC_GROUP)
    bs = pd.read_csv(BS_GROUP)

    hc = hc[hc["predictor"].isin(ORDER)].copy()
    hc["region"] = "HC"

    hc_profile = hc[[
        "predictor",
        "region",
        "beta_mean",
        "beta_sem",
    ]].copy()

    # VTA bilateral descriptive average
    vta = bs[bs["roi"].isin(["VTA_L", "VTA_R"])].copy()
    vta_profile = (
        vta.groupby("predictor", as_index=False)
        .agg(
            beta_mean=("beta_mean", "mean"),
            beta_sem=("beta_sem", "mean"),
        )
    )
    vta_profile["region"] = "VTA"

    # LC left/right separately, because they are only 1–2 voxels and exploratory
    lc = bs[bs["roi"].isin(["LC_L", "LC_R"])].copy()
    lc_profile = lc.rename(columns={"roi": "region"})[[
        "predictor",
        "region",
        "beta_mean",
        "beta_sem",
    ]].copy()

    profiles = pd.concat(
        [hc_profile, vta_profile, lc_profile],
        ignore_index=True,
    )

    profiles = profiles[profiles["predictor"].isin(ORDER)].copy()
    profiles["predictor"] = pd.Categorical(
        profiles["predictor"],
        ORDER,
        ordered=True,
    )

    return profiles.sort_values(["region", "predictor"])


def main():
    df = load_profiles()

    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    x = np.arange(len(ORDER))

    for region in ["HC", "VTA", "LC_L", "LC_R"]:
        d = df[df["region"] == region].copy()
        d = d.sort_values("predictor")

        if d.empty:
            continue

        y = d["beta_mean"].values
        err = d["beta_sem"].values

        ax.errorbar(
            x,
            y,
            yerr=err,
            marker="o",
            linewidth=2.0 if region in ["HC", "VTA"] else 1.4,
            markersize=5.5 if region in ["HC", "VTA"] else 4.5,
            capsize=3,
            label=region,
            color=COLORS[region],
            alpha=1.0 if region in ["HC", "VTA"] else 0.75,
        )

    ax.axhline(0, linewidth=1, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [LABELS[p] for p in ORDER],
        fontsize=9,
    )

    ax.set_ylabel("β", fontsize=12)
    ax.set_title(
        "Exploratory LC profile compared with VTA and hippocampus",
        fontsize=12,
        fontweight="bold",
    )

    ax.legend(frameon=False, fontsize=9, ncol=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    out_png = OUT / "supp_lc_vta_hipp_profiles.png"
    out_pdf = OUT / "supp_lc_vta_hipp_profiles.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    caption = """Supplementary Figure Sx. Exploratory locus coeruleus analyses. Mean beta profiles for hippocampus (HC), ventral tegmental area (VTA), and locus coeruleus (LC) across linguistic predictors. VTA is shown as the descriptive average of left and right VTA masks, whereas LC is shown separately for left and right masks. LC exhibited a qualitatively similar profile to HC and VTA, with weaker effects for sentence-level predictors and stronger effects for token-level transition metrics. Because the LC occupied only 1–2 functional voxels per hemisphere following atlas resampling to the functional resolution of the present dataset, these results should be interpreted as exploratory."""
    caption_file = OUT / "supp_lc_vta_hipp_profiles_caption.txt"
    caption_file.write_text(caption)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {caption_file}")


if __name__ == "__main__":
    main()