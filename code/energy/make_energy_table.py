import pandas as pd
from pathlib import Path

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"
OUT = BASE / "paper_tables"
OUT.mkdir(exist_ok=True)

# your known values
data = {
    "Metric": [
        "Mean slope (α)",
        "Slope SD (subjects/runs)",
        "R² (group fit, k=1–60)",
        "Curvature (quadratic term)",
        "Slope (k=1–40)",
        "Slope (k=1–60)",
        "Slope (k=10–80)",
        "Slope (k=20–100)",
        "Slope (k=40–120)",
    ],
    "Value": [
        -0.9967,
        0.0705,
        0.7367,
        -0.0728,
        -1.0039,
        -1.0015,
        -1.0064,
        -0.8911,
        -1.0429,
    ],
}

df = pd.DataFrame(data)

# rounding
df["Value"] = df["Value"].round(3)

out_file = OUT / "table_energy_powerlaw.csv"
df.to_csv(out_file, index=False)

print("✅ wrote", out_file)
print(df)