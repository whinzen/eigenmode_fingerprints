from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from settings import PANG_OUT, HEMIS, WAVELENGTH_TABLE

OUT_DIR = PANG_OUT / "group_sentence_shift_glm"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# FDR
# ----------------------------
def bh_fdr(p, alpha=0.05):
    p = np.asarray(p, float)
    mask = np.isfinite(p)
    sig = np.zeros(len(p), bool)

    if mask.sum() == 0:
        return sig

    idx = np.where(mask)[0]
    ps = p[mask]

    order = np.argsort(ps)
    thr = alpha * (np.arange(1, len(ps)+1) / len(ps))
    below = ps[order] <= thr

    if below.any():
        kmax = np.where(below)[0].max()
        sig[idx[order[:kmax+1]]] = True

    return sig


# ----------------------------
# Load λ table
# ----------------------------
def load_lambda_table():
    t = pd.read_csv(WAVELENGTH_TABLE)

    print("🔍 wavelength_table columns:", list(t.columns))

    # --- detect mode column ---
    for c in ["mode_k", "mode", "k"]:
        if c in t.columns:
            mode_col = c
            break
    else:
        raise RuntimeError(f"No mode column found in {t.columns}")

    # --- detect lambda column ---
    for c in ["lambda", "lam", "wavelength", "wavelength_mm"]:
        if c in t.columns:
            lam_col = c
            break
    else:
        raise RuntimeError(f"No lambda column found in {t.columns}")

    t = t.rename(columns={mode_col: "mode_k", lam_col: "lam"}).drop_duplicates("mode_k")

    return t[["mode_k", "lam"]]

# ----------------------------
# Load all per-run data
# ----------------------------
def collect_all(hemi):
    rows = []

    for sdir in sorted(PANG_OUT.glob("sub-*")):
        per = sdir / "glm_sentence_shift" / "per_run"
        if not per.is_dir():
            continue

        files = sorted(per.glob(f"sentence_shift_{hemi}_run-*.csv"))

        for f in files:
            df = pd.read_csv(f)
            rows.append(df)

    if not rows:
        raise SystemExit(f"❌ No data for hemi={hemi}")

    return pd.concat(rows, ignore_index=True)


# ----------------------------
# SUBJECT-LEVEL aggregation
# ----------------------------
def by_mode_subject_level(big, lam_table, exclude_k0=True):

    # merge λ
    big = big.merge(lam_table, on="mode_k", how="left")

    if exclude_k0:
        big = big[big["mode_k"] > 0]

    big = big[np.isfinite(big["beta"]) & np.isfinite(big["lam"])]

    # ----------------------------
    # 1. average within subject
    # ----------------------------
    subj_mean = (
        big.groupby(["subject", "mode_k"], as_index=False)
        .agg(beta=("beta", "mean"),
             lam=("lam", "median"))
    )

    # ----------------------------
    # 2. t-test across subjects
    # ----------------------------
    results = []

    for k, df_k in subj_mean.groupby("mode_k"):

        betas = df_k["beta"].values
        betas = betas[np.isfinite(betas)]

        if len(betas) < 2:
            continue

        t, p = ttest_1samp(betas, 0.0)

        results.append({
            "mode_k": k,
            "lam": df_k["lam"].median(),
            "beta_mean": np.mean(betas),
            "beta_sem": np.std(betas, ddof=1) / np.sqrt(len(betas)),
            "N_subj": len(betas),
            "t": t,
            "p": p
        })

    g = pd.DataFrame(results).sort_values("mode_k")

    # ----------------------------
    # 3. FDR
    # ----------------------------
    g["sig_q05"] = bh_fdr(g["p"].values, alpha=0.05).astype(int)

    return g


# ----------------------------
# MAIN
# ----------------------------
def main():

    lam_table = load_lambda_table()

    for hemi in HEMIS:

        big = collect_all(hemi)

        out_big = OUT_DIR / f"group_sentence_shift_allruns_hemi-{hemi}.csv"
        big.to_csv(out_big, index=False)
        print(f"✅ wrote {out_big}  rows={len(big):,}")

        g = by_mode_subject_level(big, lam_table, exclude_k0=True)

        out_sm = OUT_DIR / f"group_sentence_shift_hemi-{hemi}_by_mode_subject_level.csv"
        g.to_csv(out_sm, index=False)

        print(f"✅ wrote {out_sm}  modes={len(g)}")


if __name__ == "__main__":
    main()