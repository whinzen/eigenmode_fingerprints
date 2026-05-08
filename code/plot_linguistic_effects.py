from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out" / "word_transition_geometry"
OUT = BASE / "figures_paper"
OUT.mkdir(exist_ok=True)

CSV = BASE / "word_transition_summary.csv"

METRICS = [
    ("shift", "Shift"),
    ("pred_error_ar", "AR error"),
    ("pred_error_subspace", "Subspace exit"),
    ("curvature", "Curvature"),
]

df = pd.read_csv(CSV)

means_within = []
means_boundary = []
sems_within = []
sems_boundary = []
labels = []

for m, lab in METRICS:
    row = df[df["measure"] == m].iloc[0]

    means_within.append(row["within_mean"])
    means_boundary.append(row["boundary_mean"])
    sems_within.append(row["within_sem"])
    sems_boundary.append(row["boundary_sem"])
    labels.append(lab)

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.bar(x - width/2, means_within, width, yerr=sems_within, label="Within", capsize=4)
ax.bar(x + width/2, means_boundary, width, yerr=sems_boundary, label="Boundary", capsize=4)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Mean value")
ax.set_title("Linguistic transition metrics")
ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT / "figure_L1_linguistic_effects.png", dpi=300)
fig.savefig(OUT / "figure_L1_linguistic_effects.pdf")

print("✅ Figure L1 saved")