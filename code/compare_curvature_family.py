import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"

CURV_DIR = BASE / "group_curvature_glm"
MAIN_DIR = BASE / "group_transition_metrics"  # your original results

metrics_new = [
    "global_curvature_R",
    "mean_turning_angle",
    "path_length",
    "chord_length"
]

# original curvature
orig_file = MAIN_DIR / "group_curvature_by_mode_subject_level.csv"

def load_bihemi(f):
    df = pd.read_csv(f)

    dl = df[df["hemi"] == "L"]
    dr = df[df["hemi"] == "R"]

    merged = dl.merge(dr, on="mode_k", suffixes=("_L", "_R"))
    merged["beta"] = (merged["beta_mean_L"] + merged["beta_mean_R"]) / 2
    return merged[["mode_k", "beta"]]


orig = load_bihemi(orig_file)

results = []

for m in metrics_new:
    f = CURV_DIR / f"group_{m}_by_mode_subject_level.csv"
    df = load_bihemi(f)

    merged = orig.merge(df, on="mode_k", suffixes=("_orig", "_new"))

    # exclude mode 0
    merged = merged[merged["mode_k"] > 0]

    r = np.corrcoef(merged["beta_orig"], merged["beta_new"])[0,1]

    results.append({
        "metric": m,
        "correlation_with_original_curvature": r,
        "mean_beta_new_1_10": merged[merged["mode_k"].between(1,10)]["beta_new"].mean()
    })

res = pd.DataFrame(results)
print("\n=== Curvature family comparison ===")
print(res)

out = BASE / "paper_tables" / "table_curvature_family.csv"
out.parent.mkdir(parents=True, exist_ok=True)
res.to_csv(out, index=False)

print("✅ wrote", out)