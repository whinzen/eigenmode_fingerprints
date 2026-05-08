from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"
OUT_DIR = BASE / "group_wordrate_glm"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bh_fdr(pvals, alpha=0.05):
    pvals = np.asarray(pvals, float)
    out = np.zeros(len(pvals), dtype=int)

    good = np.isfinite(pvals)
    if good.sum() == 0:
        return out

    p = pvals[good]
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, len(ranked) + 1) / len(ranked))
    passed = ranked <= thresh

    if passed.any():
        kmax = np.where(passed)[0].max()
        keep = order[:kmax + 1]
        idx = np.where(good)[0][keep]
        out[idx] = 1

    return out


def collect():
    rows = []
    for sub_dir in sorted(BASE.glob("sub-*")):
        f = sub_dir / "glm_wordrate" / "wordrate_wide.csv"
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if len(df) == 0:
            continue
        rows.append(df)

    if not rows:
        raise RuntimeError("No wordrate_wide.csv files found. Run rebuild_glm_wordrate_per_run.py first.")

    big = pd.concat(rows, ignore_index=True)
    req = {"subject", "run", "hemi", "mode_k", "beta"}
    missing = req - set(big.columns)
    if missing:
        raise RuntimeError(f"Missing required columns {missing}. Found: {list(big.columns)}")

    return big


def main():
    big = collect()
    big.to_csv(OUT_DIR / "group_wordrate_allruns.csv", index=False)

    results = []

    for hemi, d_hemi in big.groupby("hemi"):
        subj_mean = (
            d_hemi.groupby(["subject", "mode_k"], as_index=False)
            .agg(beta_mean_subj=("beta", "mean"))
        )

        for mode_k, g in subj_mean.groupby("mode_k"):
            vals = g["beta_mean_subj"].dropna().values

            if len(vals) < 2:
                results.append({
                    "hemi": hemi,
                    "mode_k": int(mode_k),
                    "beta_mean": np.nan,
                    "beta_sem": np.nan,
                    "N_subj": len(vals),
                    "t": np.nan,
                    "p": np.nan,
                })
                continue

            t, p = ttest_1samp(vals, 0.0, nan_policy="omit")
            results.append({
                "hemi": hemi,
                "mode_k": int(mode_k),
                "beta_mean": vals.mean(),
                "beta_sem": vals.std(ddof=1) / np.sqrt(len(vals)),
                "N_subj": len(vals),
                "t": t,
                "p": p,
            })

    out = pd.DataFrame(results).sort_values(["hemi", "mode_k"]).reset_index(drop=True)

    for hemi in out["hemi"].unique():
        mask = out["hemi"] == hemi
        out.loc[mask, "sig_q05"] = bh_fdr(out.loc[mask, "p"].values, alpha=0.05)

    out["sig_q05"] = out["sig_q05"].astype(int)

    out_csv = OUT_DIR / "group_wordrate_by_mode_subject_level.csv"
    out.to_csv(out_csv, index=False)

    print(f"✅ wrote {OUT_DIR / 'group_wordrate_allruns.csv'}")
    print(f"✅ wrote {out_csv}")
    print("\nCounts by hemi:")
    print(out.groupby("hemi")["mode_k"].count())


if __name__ == "__main__":
    main()