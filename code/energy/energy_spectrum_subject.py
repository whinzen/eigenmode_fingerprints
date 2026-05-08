# energy_spectrum_subject.py

from pathlib import Path
import pandas as pd

BASE = Path.home() / "eigenmode_fingerprints"
GROUP = BASE / "pang_out" / "group"

# Load full aggregated energy data
df = pd.read_csv(GROUP / "energy_all.csv")

# Aggregate: mean over runs and hemispheres for each subject and mode
df_subject = (
    df.groupby(["subject", "mode_k"], as_index=False)
      .agg(lam=("lam", "first"), Emean=("E", "mean"))
)

# Save to CSV
out_path = GROUP / "energy_spectrum_subject.csv"
df_subject.to_csv(out_path, index=False)
print("✅ Wrote:", out_path)