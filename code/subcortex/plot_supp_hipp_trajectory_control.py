#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

IN = (
    PANG
    / "subcortex"
    / "hipp_trajectory_mean_covariate_group"
    / "group_bihemi_trajectory_trajectory_step_with_mean_covariate_token_shift.csv"
)

OUT = PANG / "subcortex" / "hipp_summary_figures"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

print("Columns:", df.columns.tolist())
print(df)

df["label"] = df["effect"].map({
    "token_shift_control_mean": "Token shift\ncontrolled for\nmean HC signal",
    "mean_signal_covariate": "Mean HC\nsignal covariate",
}).fillna(df["effect"])

fig, ax = plt.subplots(figsize=(5.0, 3.8))

x = range(len(df))

ax.bar(
    x,
    df["beta_mean"],
    yerr=df["beta_sem"],
    capsize=3,
    width=0.40,
)

ax.set_xlim(-0.7, 1.7)

ax.axhline(0, color="black", lw=1)

ax.set_xticks(list(x))
ax.set_xticklabels(df["label"], rotation=0, ha="center")

ax.set_ylabel("β predicting HC trajectory step length")
ax.set_title(
    "Token-level transitions predict reduced hippocampal\ntrajectory mobility after mean-signal control",
    fontsize=10.5,
    fontweight="bold",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

png = OUT / "supp_hipp_trajectory_step_mean_signal_control.png"
pdf = OUT / "supp_hipp_trajectory_step_mean_signal_control.pdf"

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Wrote", png)
print("Wrote", pdf)