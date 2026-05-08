# ~/eigenmode_fingerprints/code/rebuild_glm_per_run.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

from settings import EMP_DIR, PANG_OUT, BOUND_JSON, TR_SEC, HRF_SEC, MIN_TRS, HEMIS, GLM_PER_RUN_DIR

def load_boundaries(bound_json):
    with open(bound_json, "r") as f:
        return json.load(f)

def hemodynamic_shift_idx(TR, lag_sec):
    # round to nearest TRs (e.g., 6 s / 2 s = 3 TR)
    return max(0, int(round(lag_sec / TR)))

def design_from_onsets(T, TR, onsets_sec, lag_sec):
    """Binary stick convolved by simple lag (shift); then z-score."""
    idx_shift = hemodynamic_shift_idx(TR, lag_sec)
    x = np.zeros(T, dtype=float)
    for t in onsets_sec:
        i0 = int(round(t / TR)) + idx_shift
        if 0 <= i0 < T:
            x[i0] += 1.0
    # z-score if variance > 0
    if x.std() > 0:
        x = (x - x.mean()) / x.std()
    return x

def run_glm_per_run(sub_dir: Path, boundaries: dict, out_dir: Path, tr=TR_SEC, hrf=HRF_SEC):
    sub = sub_dir.name
    func_dir = sub_dir / "func"

    # Discover runs (by left hemi files) to get TR length
    lfiles = sorted(func_dir.glob(f"{sub}_task-lppEN_run-*_hemi-L_space-fsaverage5_bold.func.gii"))
    runs = [int(f.name.split("_run-")[1].split("_")[0]) for f in lfiles]

    out_dir.mkdir(parents=True, exist_ok=True)
    wide_rows = []

    for run in sorted(runs):
        # locate energy time-series per hemisphere (already computed by your eigenmode projection step)
        # Expect: pang_out/sub-XXX/run-YY/E_k_t_hemi-H.npy (shape [K,T])
        ekL = PANG_OUT / sub / f"run-{run}" / "E_k_t_hemi-L.npy"
        ekR = PANG_OUT / sub / f"run-{run}" / "E_k_t_hemi-R.npy"
        if not ekL.exists() or not ekR.exists():
            print(f"[skip] {sub} run-{run}: missing E_k_t .npy")
            continue

        EkL = np.load(ekL)   # (K,T)
        EkR = np.load(ekR)
        K, T = EkL.shape
        if T < MIN_TRS:
            print(f"[skip] {sub} run-{run}: T={T} < {MIN_TRS}")
            continue

        # build design from per-run sentence ends (offsets)
        key = f"run-{run}"
        onsets = boundaries.get(key, [])
        x = design_from_onsets(T, tr, onsets, hrf)  # (T,)

        X = sm.add_constant(x)  # [T,2] with intercept

        for H, Ek in zip(["L","R"], [EkL, EkR]):
            betas, pvals = [], []
            for k in range(K):
                y = Ek[k].astype(float)
                if np.all(~np.isfinite(y)) or y.std() == 0:
                    betas.append(np.nan); pvals.append(np.nan); continue
                model = sm.OLS(y, X, missing="drop").fit()
                # Regressor of interest is x (column 1)
                betas.append(model.params[1] if len(model.params) > 1 else np.nan)
                pvals.append(model.pvalues[1] if len(model.pvalues) > 1 else np.nan)

            df = pd.DataFrame({
                "subject": sub, "run": run, "hemi": H,
                "mode_k": np.arange(K, dtype=int),
                "beta": betas, "p": pvals
            })

            # per-run, per-hemi file
            out_csv = out_dir / f"onset_{H}_run-{run}.csv"
            df.to_csv(out_csv, index=False)

            # add to wide report
            for k, b, p in zip(df["mode_k"], df["beta"], df["p"]):
                wide_rows.append({"subject": sub, "run": run, "hemi": H, "mode_k": int(k), "beta": b, "p": p})

    # per-subject wide (all runs & hemis)
    wide = pd.DataFrame(wide_rows)
    wide_out = sub_dir / "glm_sentence" / "onset_wide.csv"
    wide_out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(wide_out, index=False)
    print(f"[{sub}] wrote {wide_out} and per-run onset_<H>_run-*.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", help="Optional subset like sub-EN0xx ...")
    args = ap.parse_args()

    with open(BOUND_JSON, "r") as f:
        boundaries = json.load(f)

    subj_dirs = [d for d in sorted(EMP_DIR.glob("sub-*")) if d.is_dir()]
    if args.subjects:
        keep = set(args.subjects)
        subj_dirs = [d for d in subj_dirs if d.name in keep]

    for sdir in subj_dirs:
        out_dir = PANG_OUT / sdir.name / "glm_sentence" / "per_run"
        run_glm_per_run(sdir, boundaries, out_dir)

if __name__ == "__main__":
    main()