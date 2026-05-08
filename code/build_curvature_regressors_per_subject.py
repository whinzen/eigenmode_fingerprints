import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path.home() / "eigenmode_fingerprints"
PANG_OUT = BASE / "pang_out"

CSV_PATH = BASE / "ds003643" / "annotation" / "EN" / "repunct" / "little_prince_series_curvature.csv"

OUT_DIR = PANG_OUT / "regressors" / "curvature_per_subject"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TR = 2.0


def zscore_safe(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return x
    s = x.std()
    if s == 0:
        return x - x.mean()
    return (x - x.mean()) / s


def get_subject_runs(sub_dir):
    runs = []
    for rdir in sorted(sub_dir.glob("run-*")):
        try:
            runs.append(int(rdir.name.split("-")[1]))
        except:
            continue
    return sorted(runs)


def get_run_length(sub_dir, run):
    f = sub_dir / f"run-{run:02d}" / "A_L.npy"
    if not f.exists():
        return None
    return np.load(f).shape[1]


def build_tr_regressor(df_run, col):
    tmax = df_run["onset"].max()
    T = int(np.ceil(tmax / TR)) + 1
    x = np.zeros(T)

    for _, row in df_run.iterrows():
        t = row["onset"]
        val = row[col]
        idx = int(np.floor(t / TR))
        if 0 <= idx < T:
            x[idx] += val

    return zscore_safe(x)


def main():
    df = pd.read_csv(CSV_PATH)

    curvature_cols = [
        "global_curvature_R",
        "mean_turning_angle",
        "path_length",
        "chord_length"
    ]

    print("Using curvature metrics:", curvature_cols)

    # build stimulus-space regressors
    stim_regs = {col: {} for col in curvature_cols}

    for run_id, g in df.groupby("run_id"):
        g = g.sort_values("onset")

        for col in curvature_cols:
            x = build_tr_regressor(g, col)
            stim_regs[col][run_id] = x

    subjects = sorted([p for p in PANG_OUT.glob("sub-*") if p.is_dir()])
    stim_runs = sorted(df["run_id"].unique())

    for sub_dir in subjects:
        sub = sub_dir.name
        subj_runs = get_subject_runs(sub_dir)
        n = min(len(subj_runs), len(stim_runs))

        for i in range(n):
            subj_run = subj_runs[i]
            stim_run = stim_runs[i]

            T_actual = get_run_length(sub_dir, subj_run)
            if T_actual is None:
                continue

            for col in curvature_cols:
                x = stim_regs[col][stim_run].copy()

                if len(x) > T_actual:
                    x = x[:T_actual]
                elif len(x) < T_actual:
                    x = np.pad(x, (0, T_actual - len(x)))

                x = zscore_safe(x)

                out = OUT_DIR / f"{sub}_run-{subj_run:02d}_{col}.npy"
                np.save(out, x)

        print(f"✅ {sub} done")


if __name__ == "__main__":
    main()