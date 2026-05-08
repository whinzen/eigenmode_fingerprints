import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints"
PANG_OUT = BASE / "pang_out"
WAVELENGTH_TABLE = PANG_OUT / "group" / "wavelength_table.csv"

def bh_fdr(p, alpha=0.05):
    p = np.asarray(p, float)
    sig = np.zeros(len(p), dtype=bool)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return sig
    idx = np.where(mask)[0]
    ps = p[mask]
    order = np.argsort(ps)
    thr = alpha * (np.arange(1, len(ps) + 1) / len(ps))
    below = ps[order] <= thr
    if below.any():
        kmax = np.where(below)[0].max()
        sig[idx[order[:kmax + 1]]] = True
    return sig

def load_lambda_table():
    t = pd.read_csv(WAVELENGTH_TABLE)
    if "mode_k" not in t.columns and "k" in t.columns:
        t = t.rename(columns={"k": "mode_k"})
    if "lam" not in t.columns and "lambda" in t.columns:
        t = t.rename(columns={"lambda": "lam"})
    if "mode_k" not in t.columns or "lam" not in t.columns:
        raise RuntimeError(f"wavelength table missing required columns. Found: {list(t.columns)}")
    return t[["mode_k", "lam"]].drop_duplicates("mode_k")

def collect_all_runs(pair_name, hemi):
    rows = []
    for sdir in sorted(PANG_OUT.glob("sub-*")):
        per = sdir / f"glm_{pair_name}" / "per_run"
        if not per.exists():
            continue
        for f in sorted(per.glob(f"{pair_name}_{hemi}_run-*.csv")):
            df = pd.read_csv(f)
            df["source_file"] = str(f)
            rows.append(df)
    if not rows:
        raise SystemExit(f"❌ no per-run files for pair={pair_name}, hemi={hemi}")
    return pd.concat(rows, ignore_index=True)

def aggregate_subject_level(big, lam_table, beta_col, exclude_k0=True):
    if exclude_k0:
        big = big[big["mode_k"] > 0].copy()

    subj_mean = (
        big.groupby(["subject", "mode_k"], as_index=False)
        .agg(beta=(beta_col, "mean"))
    )

    out = []
    for mode_k, g in subj_mean.groupby("mode_k"):
        vals = g["beta"].dropna().values
        if len(vals) < 2:
            out.append({
                "mode_k": int(mode_k),
                "beta_mean": np.nan,
                "beta_sem": np.nan,
                "N_subj": len(vals),
                "t": np.nan,
                "p": np.nan,
            })
            continue

        t, p = ttest_1samp(vals, 0.0, nan_policy="omit")
        out.append({
            "mode_k": int(mode_k),
            "beta_mean": vals.mean(),
            "beta_sem": vals.std(ddof=1) / np.sqrt(len(vals)),
            "N_subj": len(vals),
            "t": t,
            "p": p,
        })

    out = pd.DataFrame(out)
    out = out.merge(lam_table, on="mode_k", how="left")
    out["sig_q05"] = bh_fdr(out["p"].values, alpha=0.05).astype(int)
    return out.sort_values("mode_k")

def plot_pair(g_base, g_extra, out_png, title, base_metric, extra_metric):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(g_base["mode_k"], g_base["beta_mean"], label=base_metric)
    ax.fill_between(
        g_base["mode_k"],
        g_base["beta_mean"] - g_base["beta_sem"],
        g_base["beta_mean"] + g_base["beta_sem"],
        alpha=0.2
    )

    ax.plot(g_extra["mode_k"], g_extra["beta_mean"], label=f"{extra_metric}_resid")
    ax.fill_between(
        g_extra["mode_k"],
        g_extra["beta_mean"] - g_extra["beta_sem"],
        g_extra["beta_mean"] + g_extra["beta_sem"],
        alpha=0.2
    )

    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Mode k")
    ax.set_ylabel("Beta")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"✅ wrote {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_metric", required=True)
    ap.add_argument("--extra_metric", required=True)
    args = ap.parse_args()

    base_metric = args.base_metric
    extra_metric = args.extra_metric
    pair_name = f"{base_metric}__plus__{extra_metric}_resid"

    lam_table = load_lambda_table()
    out_dir = PANG_OUT / f"group_{pair_name}_glm"
    out_dir.mkdir(parents=True, exist_ok=True)

    beta_base_col = f"beta_{base_metric}"
    beta_extra_col = f"beta_{extra_metric}_resid"

    for hemi in ["L", "R"]:
        big = collect_all_runs(pair_name, hemi)
        big_out = out_dir / f"group_{pair_name}_allruns_hemi-{hemi}.csv"
        big.to_csv(big_out, index=False)

        g_base = aggregate_subject_level(big, lam_table, beta_base_col, exclude_k0=True)
        g_extra = aggregate_subject_level(big, lam_table, beta_extra_col, exclude_k0=True)

        g_base_out = out_dir / f"group_{pair_name}_{base_metric}_hemi-{hemi}_by_mode_subject_level.csv"
        g_extra_out = out_dir / f"group_{pair_name}_{extra_metric}_resid_hemi-{hemi}_by_mode_subject_level.csv"

        g_base.to_csv(g_base_out, index=False)
        g_extra.to_csv(g_extra_out, index=False)

        plot_pair(
            g_base,
            g_extra,
            out_dir / f"group_{pair_name}_hemi-{hemi}_comparison.png",
            title=f"{pair_name} hemi-{hemi}",
            base_metric=base_metric,
            extra_metric=extra_metric
        )

        print(f"✅ wrote {big_out}")
        print(f"✅ wrote {g_base_out}")
        print(f"✅ wrote {g_extra_out}")

if __name__ == "__main__":
    main()