from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"
OUT_DIR = BASE / "group_joint_wordrate_resid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "shift",
    "pred_error_ar",
    "pred_error_subspace",
    "curvature",
]


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


def aggregate_one(metric):
    rows = []

    print(f"\n=== Aggregating residualized: {metric} ===")

    for sub_dir in sorted(BASE.glob("sub-*")):
        f = sub_dir / f"glm_joint_wordrate_resid_{metric}" / f"joint_wordrate_resid_{metric}_wide.csv"
        if not f.exists():
            continue
        if f.stat().st_size == 0:
            print(f"[empty file skipped] {f}")
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[read error skipped] {f} → {e}")
            continue
        if len(df) == 0:
            print(f"[no rows skipped] {f}")
            continue
        rows.append(df)

    if not rows:
        raise RuntimeError(f"No usable residualized joint files found for {metric}")

    big = pd.concat(rows, ignore_index=True)
    big.to_csv(OUT_DIR / f"group_joint_wordrate_resid_{metric}_allruns.csv", index=False)

    out_rows = []

    for hemi, d_hemi in big.groupby("hemi"):
        subj_mean = (
            d_hemi.groupby(["subject", "mode_k"], as_index=False)
            .agg(
                beta_wordrate_subj=("beta_wordrate", "mean"),
                beta_metric_resid_subj=("beta_metric_resid", "mean"),
            )
        )

        for mode_k, g in subj_mean.groupby("mode_k"):
            wr = g["beta_wordrate_subj"].dropna().values
            met = g["beta_metric_resid_subj"].dropna().values

            if len(wr) >= 2:
                t_wr, p_wr = ttest_1samp(wr, 0.0, nan_policy="omit")
                mean_wr = wr.mean()
                sem_wr = wr.std(ddof=1) / np.sqrt(len(wr))
            else:
                t_wr = p_wr = mean_wr = sem_wr = np.nan

            if len(met) >= 2:
                t_met, p_met = ttest_1samp(met, 0.0, nan_policy="omit")
                mean_met = met.mean()
                sem_met = met.std(ddof=1) / np.sqrt(len(met))
            else:
                t_met = p_met = mean_met = sem_met = np.nan

            out_rows.append({
                "metric": metric,
                "hemi": hemi,
                "mode_k": int(mode_k),
                "beta_wordrate_mean": mean_wr,
                "beta_wordrate_sem": sem_wr,
                "p_wordrate": p_wr,
                "beta_metric_resid_mean": mean_met,
                "beta_metric_resid_sem": sem_met,
                "p_metric_resid": p_met,
                "N_subj": len(g),
            })

    out = pd.DataFrame(out_rows).sort_values(["hemi", "mode_k"]).reset_index(drop=True)

    for hemi in out["hemi"].unique():
        mask = out["hemi"] == hemi
        out.loc[mask, "sig_wordrate_q05"] = bh_fdr(out.loc[mask, "p_wordrate"].values)
        out.loc[mask, "sig_metric_resid_q05"] = bh_fdr(out.loc[mask, "p_metric_resid"].values)

    out["sig_wordrate_q05"] = out["sig_wordrate_q05"].astype(int)
    out["sig_metric_resid_q05"] = out["sig_metric_resid_q05"].astype(int)

    out_csv = OUT_DIR / f"group_joint_wordrate_resid_{metric}_by_mode_subject_level.csv"
    out.to_csv(out_csv, index=False)
    print(f"✅ wrote {out_csv}")


def main():
    for metric in METRICS:
        aggregate_one(metric)


if __name__ == "__main__":
    main()