from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"
OUT_DIR = BASE / "group_curvature_glm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["global", "mean", "path", "chord"]


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


def collect(metric):
    rows = []

    # robust search: any subject folder containing the expected wide file
    files = sorted(BASE.glob(f"sub-*/glm_{metric}/{metric}_wide.csv"))

    print(f"[{metric}] found {len(files)} wide files")

    if len(files) == 0:
        # helpful diagnostics
        examples = sorted(BASE.glob("sub-*/glm_*/*.csv"))[:20]
        print("Example CSVs found:")
        for e in examples:
            print("  ", e)
        raise RuntimeError(f"No {metric}_wide.csv files found")

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[skip read error] {f}: {e}")
            continue

        if len(df) == 0:
            print(f"[skip empty] {f}")
            continue

        rows.append(df)

    if not rows:
        raise RuntimeError(f"No usable rows for metric={metric}")

    big = pd.concat(rows, ignore_index=True)

    req = {"subject", "run", "hemi", "mode_k", "beta"}
    missing = req - set(big.columns)
    if missing:
        raise RuntimeError(f"Missing columns {missing}. Found: {list(big.columns)}")

    return big


def main():
    for metric in METRICS:
        print(f"\n=== Aggregating: {metric} ===")
        big = collect(metric)
        big.to_csv(OUT_DIR / f"group_{metric}_allruns.csv", index=False)

        rows = []
        for hemi, d_hemi in big.groupby("hemi"):
            subj_mean = (
                d_hemi.groupby(["subject", "mode_k"], as_index=False)
                .agg(beta_mean_subj=("beta", "mean"))
            )

            for mode_k, g in subj_mean.groupby("mode_k"):
                vals = g["beta_mean_subj"].dropna().values

                if len(vals) < 2:
                    rows.append({
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
                rows.append({
                    "hemi": hemi,
                    "mode_k": int(mode_k),
                    "beta_mean": vals.mean(),
                    "beta_sem": vals.std(ddof=1) / np.sqrt(len(vals)),
                    "N_subj": len(vals),
                    "t": t,
                    "p": p,
                })

        out = pd.DataFrame(rows).sort_values(["hemi", "mode_k"]).reset_index(drop=True)

        for hemi in out["hemi"].unique():
            mask = out["hemi"] == hemi
            out.loc[mask, "sig_q05"] = bh_fdr(out.loc[mask, "p"].values, alpha=0.05)

        out["sig_q05"] = out["sig_q05"].astype(int)

        out_csv = OUT_DIR / f"group_{metric}_by_mode_subject_level.csv"
        out.to_csv(out_csv, index=False)
        print(f"✅ wrote {out_csv}")


if __name__ == "__main__":
    main()