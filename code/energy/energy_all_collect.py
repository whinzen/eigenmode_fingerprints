# energy_all_collect.py

from pathlib import Path
import pandas as pd

BASE = Path.home() / "eigenmode_fingerprints"
OUT = BASE / "pang_out"
GROUP = OUT / "group"
GROUP.mkdir(parents=True, exist_ok=True)

def aggregate_energy():
    rows = []
    for sub_dir in sorted(OUT.glob("sub-*")):
        subject = sub_dir.name
        for run_dir in sorted(sub_dir.glob("run-*")):
            run = run_dir.name.split("run-")[-1]
            for hemi in ["L", "R"]:
                fpath = run_dir / f"energy_{hemi}.csv"
                if fpath.exists():
                    df = pd.read_csv(fpath)
                    df["subject"] = subject
                    df["run"] = run
                    df["hemi"] = hemi
                    rows.append(df)
    if rows:
        df_all = pd.concat(rows, ignore_index=True)
        df_all.to_csv(GROUP / "energy_all.csv", index=False)
        print(f"✅ Aggregated energy file written: {GROUP / 'energy_all.csv'}")
    else:
        print("❌ No energy files found. Please check pang_out/sub-*/run-*/energy_*.csv")

if __name__ == "__main__":
    aggregate_energy()