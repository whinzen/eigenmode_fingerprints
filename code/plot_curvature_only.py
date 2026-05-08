import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"
CURV_DIR = BASE / "group_curvature_glm"
OUT_DIR = BASE / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

metrics = {
    "global": "Global curvature",
    "mean": "Mean turning angle",
    "path": "Path length",
    "chord": "Chord length",
}

plt.style.use("default")

fig, ax = plt.subplots(figsize=(7.5, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")


def load_bihemi(f):
    df = pd.read_csv(f)

    dl = df[df["hemi"] == "L"]
    dr = df[df["hemi"] == "R"]

    merged = dl.merge(dr, on="mode_k", suffixes=("_L", "_R"))
    merged["beta"] = (merged["beta_mean_L"] + merged["beta_mean_R"]) / 2

    return merged


for metric_key, metric_label in metrics.items():
    f = CURV_DIR / f"group_{metric_key}_by_mode_subject_level.csv"
    if not f.exists():
        print("⚠️ missing:", f)
        continue

    df = load_bihemi(f)
    df = df[df["mode_k"] > 0]

    ax.plot(
        df["mode_k"],
        df["beta"],
        linewidth=2,
        label=metric_label
    )

ax.axhline(0, color="gray", linewidth=1)
ax.set_xlabel("Eigenmode index (k)")
ax.set_ylabel("Mean β")
ax.set_title("Eigenmode profiles of new curvature metrics")
ax.legend(frameon=False)
ax.grid(False)

plt.tight_layout()

out_png = OUT_DIR / "figure_curvature_only.png"
out_pdf = OUT_DIR / "figure_curvature_only.pdf"

plt.savefig(out_png, dpi=300, facecolor="white")
plt.savefig(out_pdf, dpi=300, facecolor="white")
plt.close()

print("✅ wrote", out_png)
print("✅ wrote", out_pdf)