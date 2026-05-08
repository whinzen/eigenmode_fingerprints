import pandas as pd
from pathlib import Path

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

df = pd.read_csv(PANG / "paper_tables/table_bert_vs_qwen_comparison.csv")

df = df[[
    "metric", "label", "hemi",
    "pearson_r", "spearman_r", "cosine_similarity",
    "z_profile_r", "slope", "intercept"
]].copy()

df = df.rename(columns={
    "pearson_r": "r",
    "spearman_r": "rho",
    "cosine_similarity": "cosine",
    "z_profile_r": "r_z",
})

for c in ["r", "rho", "cosine", "r_z", "slope", "intercept"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").round(3)

out = PANG / "paper_tables/table_bert_vs_qwen_combined_excel_locale.csv"

df.to_csv(
    out,
    index=False,
    sep=";",
    decimal=",",
)

print("wrote:", out)
print(df.to_string(index=False))