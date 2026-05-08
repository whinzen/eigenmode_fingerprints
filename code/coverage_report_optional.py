# ~/eigenmode_fingerprints/code/coverage_report.py
from pathlib import Path
import pandas as pd
from settings import PANG_OUT, HEMIS

BASE = PANG_OUT / "group_boundary_glm"

def main():
    for hemi in HEMIS:
        big = pd.read_csv(BASE/f"group_onset_perrun_hemi-{hemi}.csv")
        subj = big.groupby("subject")["run"].nunique().sort_index()
        print(f"\n{hemi} runs per subject:\n{subj.to_string()}")
        per_mode_N = big.groupby("mode_k")["beta"].count()
        print(f"\n{hemi} per-mode N rows:\n{per_mode_N.head(10)} ... (total modes: {per_mode_N.shape[0]})")

if __name__ == "__main__":
    main()