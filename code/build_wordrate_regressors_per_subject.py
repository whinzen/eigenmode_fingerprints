import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path.home() / "eigenmode_fingerprints"
PANG_OUT = BASE / "pang_out"
CSV_PATH = BASE / "ds003643" / "annotation" / "EN" / "repunct" / "lppEN.csv"
OUT_DIR = PANG_OUT / "regressors" / "wordrate_per_subject"
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


def get_subject_runs(sub_dir: Path):
    runs = []
    for rdir in sorted(sub_dir.glob("run-*")):
        if not rdir.is_dir():
            continue
        try:
            run = int(rdir.name.split("-")[1])
            runs.append(run)
        except Exception:
            continue
    return sorted(runs)


def get_run_length_from_A(sub_dir: Path, run: int):
    a_path = sub_dir / f"run-{run:02d}" / "A_L.npy"
    if not a_path.exists():
        return None
    A = np.load(a_path)
    if A.ndim != 2:
        return None
    return A.shape[1]


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing transcript CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    possible_onset_cols = ["onset", "onset_sec", "word_onset", "start"]
    onset_col = None
    for c in possible_onset_cols:
        if c in df.columns:
            onset_col = c
            break
    if onset_col is None:
        raise RuntimeError(
            f"Could not find onset column in {CSV_PATH}. Found: {list(df.columns)}"
        )

    if "run_id" not in df.columns:
        raise RuntimeError(f"Missing run_id column in {CSV_PATH}")

    df = df.copy()
    df["run_id"] = df["run_id"].astype(int)
    df[onset_col] = df[onset_col].astype(float)

    # stimulus-space wordrate by annotation run
    stim_regs = {}
    for run_id, g in df.groupby("run_id", sort=True):
        g = g.sort_values(onset_col).reset_index(drop=True)

        tmax = g[onset_col].max()
        T = int(np.ceil(tmax / TR)) + 1
        x = np.zeros(T, dtype=float)

        for t in g[onset_col].values:
            idx = int(np.floor(t / TR))
            if 0 <= idx < T:
                x[idx] += 1.0

        stim_regs[int(run_id)] = zscore_safe(x)

    print(f"✅ Built stimulus-space wordrate regressors for runs: {sorted(stim_regs.keys())}")

    subjects = sorted([p for p in PANG_OUT.glob("sub-*") if p.is_dir()])
    stim_run_ids = sorted(stim_regs.keys())

    for sub_dir in subjects:
        sub = sub_dir.name
        subj_runs = get_subject_runs(sub_dir)
        n = min(len(subj_runs), len(stim_run_ids))

        if n == 0:
            print(f"[skip] {sub}: no runs found")
            continue

        written = 0

        for i in range(n):
            subj_run = subj_runs[i]
            stim_run = stim_run_ids[i]

            x = stim_regs[stim_run].copy()

            T_actual = get_run_length_from_A(sub_dir, subj_run)
            if T_actual is None:
                print(f"[skip] {sub} run-{subj_run:02d}: missing A_L.npy")
                continue

            if len(x) > T_actual:
                x = x[:T_actual]
            elif len(x) < T_actual:
                pad = np.zeros(T_actual - len(x), dtype=float)
                x = np.concatenate([x, pad])

            x = zscore_safe(x)

            out = OUT_DIR / f"{sub}_run-{subj_run:02d}.npy"
            np.save(out, x)
            written += 1

        print(f"✅ {sub}: wrote {written} subject-specific wordrate regressors")


if __name__ == "__main__":
    main()