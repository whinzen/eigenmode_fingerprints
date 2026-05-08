import pandas as pd
import numpy as np
from pathlib import Path

# -------- Paths --------
BASE = Path.home() / "eigenmode_fingerprints"
IN_DIR = BASE / "pang_out" / "group"
OUT_DIR = IN_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE_GROUP = OUT_DIR / "group_temporal_scaling.csv"
OUT_FILE_SEM   = OUT_DIR / "group_temporal_scaling_SEM.csv"
# -----------------------

print("🔍 Searching for temporal scaling files...")

all_dfs = []

for f in sorted(IN_DIR.glob("sub-*_run-*_temporal_scaling.csv")):
    df = pd.read_csv(f)
    all_dfs.append(df)

if not all_dfs:
    raise SystemExit("❌ No temporal scaling files found.")

print(f"✔ Loaded {len(all_dfs)} files.")

df_all = pd.concat(all_dfs, ignore_index=True)
print(f"✔ Combined dataframe shape: {df_all.shape}")

# Group by subject and vertex: average across runs per subject
df_subj_mean = (
    df_all.groupby(["subject", "vertex"], as_index=False)
          .agg(
              slope_alpha=("slope_alpha", "mean"),
              fractal_dim=("fractal_dim", "mean")
          )
)

# Save per-subject averages
df_subj_mean.to_csv(OUT_DIR / "subject_temporal_scaling.csv", index=False)

# Group average across subjects (mean and SEM)
df_group = (
    df_subj_mean.groupby("vertex", as_index=False)
        .agg(
            slope_alpha=("slope_alpha", "mean"),
            fractal_dim=("fractal_dim", "mean")
        )
)

df_sem = (
    df_subj_mean.groupby("vertex", as_index=False)
        .agg(
            slope_alpha_SEM=("slope_alpha", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            fractal_dim_SEM=("fractal_dim", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        )
)

# Merge mean and SEM
df_out = pd.merge(df_group, df_sem, on="vertex")

# Save
df_out.to_csv(OUT_FILE_GROUP, index=False)
print("✅ Group-level temporal scaling written to:")
print("  ", OUT_FILE_GROUP)