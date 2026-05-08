python - <<'PY'
import os, math
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import linregress, ttest_1samp

# ---------- config ----------
BASE      = Path.home() / "eigenmode_fingerprints"
EMP_ROOT  = BASE / "data/empirical"
CSV_PATH  = BASE / "ds003643/annotation/EN/repunct/lppEN.csv"
MODES_DIR = BASE / "data/template_eigenmodes/fsaverage5"
OUT_ROOT  = BASE / "pang_out"
TR_SEC    = 2.0   # LPP fMRI TR; change if needed

# ---------- helpers ----------
def load_func_gii(path):
    g = nib.load(str(path))
    X = np.column_stack([da.data for da in g.darrays]).astype(float)  # (V, T)
    return X

def standardize_rows(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd==0] = 1.0
    return (X - mu) / sd

def build_stick(run_events_s, T):
    """events in seconds → TR bins [0..T-1], stick(1.0 at event TR)."""
    idx = np.clip(np.round(np.array(run_events_s)/TR_SEC).astype(int), 0, T-1)
    v = np.zeros(T, float)
    if idx.size:
        v[np.unique(idx)] = 1.0
    return v

def best_run_offset(disk_runs, csv_runs):
    """Find integer offset o s.t. csv_run+o overlaps disk_runs maximally."""
    if not disk_runs or not csv_runs:
        return 0, []
    disk_set = set(disk_runs)
    best_o, best_overlap, best_shifted = 0, -1, []
    for o in range(-20, 21):  # generous search
        shifted = [int(r+o) for r in csv_runs]
        overlap = len([r for r in shifted if r in disk_set])
        if overlap > best_overlap:
            best_overlap = overlap
            best_o = o
            best_shifted = shifted
    return best_o, best_shifted

def bh_fdr(p, alpha=0.05):
    p = np.asarray(p)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n+1) / n)
    passed = ranked <= thresh
    k = np.max(np.where(passed)[0]) + 1 if np.any(passed) else 0
    cutoff = ranked[k-1] if k>0 else 0.0
    q = np.ones_like(p, dtype=float)
    # standard BH adjusted q-values
    # q_i = min_{j>=i} n/j * p_(j)  (in order), then map back
    q_order = np.minimum.accumulate((n/np.arange(1, n+1)) * ranked[::-1])[::-1]
    q[order] = np.minimum(q_order, 1.0)
    sig = p <= cutoff if k>0 else np.zeros_like(p, dtype=bool)
    return q, sig

# ---------- load template eigenmodes ----------
phi_L = np.load(MODES_DIR/"phi_L.npy")  # (V, K)
phi_R = np.load(MODES_DIR/"phi_R.npy")
lam_L = np.load(MODES_DIR/"lam_L.npy")  # (K,)
lam_R = np.load(MODES_DIR/"lam_R.npy")
K = phi_L.shape[1]

# ---------- load CSV & prepare sentence events ----------
csv = pd.read_csv(CSV_PATH)
# columns expected: snt_id, token_id, onset, offset, run_id (onset/offset in milliseconds)
csv = csv.rename(columns={c:c.strip() for c in csv.columns})
for col in ["snt_id","token_id","onset","offset","run_id"]:
    if col not in csv.columns:
        raise RuntimeError(f"Missing column '{col}' in CSV {CSV_PATH}")

# group events by run_id (1..9)
runs_csv = sorted(csv["run_id"].unique().tolist())

# ---------- per-subject processing ----------
subs = sorted([p for p in EMP_ROOT.iterdir() if p.is_dir()])
sub_betas = {"onset":{"L":[], "R":[]}, "offset":{"L":[], "R":[]}}
sub_ids   = []

for sub_dir in subs:
    sub = sub_dir.name
    func_dir = sub_dir/"func"
    if not func_dir.exists():
        continue

    # find L/R func files and extract run nums
    funcL = sorted(func_dir.glob(f"{sub}_task-lppEN_run-*_hemi-L_space-fsaverage5_bold.func.gii"))
    funcR = sorted(func_dir.glob(f"{sub}_task-lppEN_run-*_hemi-R_space-fsaverage5_bold.func.gii"))
    if not funcL or not funcR:
        print(f"[skip] {sub}: missing fsaverage5 func gii")
        continue

    # parse run numbers from filenames
    def parse_run(p):
        # ..._run-15_...
        for part in p.name.split("_"):
            if part.startswith("run-"):
                return int(part.split("-")[1])
        return None
    disk_runs = sorted({parse_run(p) for p in funcL})
    if None in disk_runs:
        disk_runs.remove(None)

    # best offset: CSV run_id -> disk run numbers
    o, shifted = best_run_offset(disk_runs, runs_csv)
    csv_to_disk = {int(r):int(r+o) for r in runs_csv}
    # mapping func path dicts
    L_by_run = {parse_run(p):p for p in funcL}
    R_by_run = {parse_run(p):p for p in funcR}

    # collect per-run time series (concatenate across runs)
    # and build two regressors (onset/offset)
    E_L_concat = []
    E_R_concat = []
    X_on_concat = []
    X_off_concat = []

    for rid_csv in runs_csv:
        rid_disk = csv_to_disk[rid_csv]
        if rid_disk not in L_by_run or rid_disk not in R_by_run:
            continue

        XL = standardize_rows(load_func_gii(L_by_run[rid_disk]))
        XR = standardize_rows(load_func_gii(R_by_run[rid_disk]))
        # activations & energy
        AL = phi_L.T @ XL  # (K,T)
        AR = phi_R.T @ XR
        EL = (AL**2)       # (K,T)
        ER = (AR**2)

        T = EL.shape[1]

        # collect sentence events for this CSV run
        df_run = csv[csv["run_id"]==rid_csv]

        # Onsets: token_id==1
        onset_ms = df_run.loc[df_run["token_id"]==1, "onset"].values
        onset_s  = onset_ms / 1000.0
        X_on = build_stick(onset_s, T)

        # Offsets: last token in each sentence (max token_id per snt_id)
        last = df_run.sort_values(["snt_id","token_id"]).groupby("snt_id").tail(1)
        offset_ms = last["offset"].values
        offset_s  = offset_ms / 1000.0
        X_off = build_stick(offset_s, T)

        # append
        E_L_concat.append(EL)
        E_R_concat.append(ER)
        X_on_concat.append(X_on)
        X_off_concat.append(X_off)

    if not E_L_concat:
        print(f"[warn] {sub}: no overlapping runs after alignment (offset +{o})")
        continue

    # concatenate over time
    E_L = np.concatenate(E_L_concat, axis=1)   # (K, sumT)
    E_R = np.concatenate(E_R_concat, axis=1)
    X_on  = np.concatenate(X_on_concat)        # (sumT,)
    X_off = np.concatenate(X_off_concat)

    # demean energy per mode (common in GLMs)
    def demean_rows(Y):
        return Y - Y.mean(axis=1, keepdims=True)

    E_Lz = demean_rows(E_L)
    E_Rz = demean_rows(E_R)

    # Run OLS per mode & hemi: y = a + b * X
    def regress_all(E, X):
        betas, pvals = [], []
        for k in range(E.shape[0]):
            y = E[k]
            res = linregress(X, y)  # slope is our beta
            betas.append(res.slope)
            pvals.append(res.pvalue)
        return np.array(betas), np.array(pvals)

    # ONSETS
    bL_on, pL_on = regress_all(E_Lz, X_on)
    bR_on, pR_on = regress_all(E_Rz, X_on)
    # OFFSETS
    bL_off, pL_off = regress_all(E_Lz, X_off)
    bR_off, pR_off = regress_all(E_Rz, X_off)

    # save subject-level CSVs
    sub_out = OUT_ROOT / sub / "glm_sentence"
    sub_out.mkdir(parents=True, exist_ok=True)
    for name, bL, pL, bR, pR in [
        ("onset",  bL_on,  pL_on,  bR_on,  pR_on),
        ("offset", bL_off, pL_off, bR_off, pR_off),
    ]:
        df_sub = pd.DataFrame({
            "mode_k": np.arange(K, dtype=int),
            "lam_L": lam_L,
            "lam_R": lam_R,
            "beta_L": bL, "p_L": pL,
            "beta_R": bR, "p_R": pR,
        })
        df_sub.to_csv(sub_out/f"{name}.csv", index=False)

    # stash for group
    sub_betas["onset"]["L"].append(bL_on)
    sub_betas["onset"]["R"].append(bR_on)
    sub_betas["offset"]["L"].append(bL_off)
    sub_betas["offset"]["R"].append(bR_off)
    sub_ids.append(sub)
    print(f"[OK] {sub}: runs aligned with offset +{o}. Saved subject GLMs to {sub_out}")

# ---------- GROUP summary ----------
if not sub_ids:
    raise SystemExit("No subjects completed; nothing to summarize.")

group_dir = OUT_ROOT / "group_glm_sentence"
group_dir.mkdir(parents=True, exist_ok=True)

def summarize(betas_list):
    B = np.stack(betas_list, axis=0)  # (Nsub, K)
    mean = B.mean(axis=0)
    sem  = B.std(axis=0, ddof=1) / math.sqrt(B.shape[0])
    # one-sample t-test vs 0 per mode
    tvals, pvals = ttest_1samp(B, 0.0, axis=0, alternative="two-sided", nan_policy="omit")
    qvals, sig = bh_fdr(pvals, alpha=0.05)
    return mean, sem, pvals, qvals, sig

for name in ["onset","offset"]:
    for hemi in ["L","R"]:
        mean, sem, pvals, qvals, sig = summarize(sub_betas[name][hemi])
        df = pd.DataFrame({
            "mode_k": np.arange(K, dtype=int),
            "lambda": lam_L if hemi=="L" else lam_R,
            "beta_mean": mean,
            "beta_sem": sem,
            "p": pvals,
            "q": qvals,
            "sig_q05": sig.astype(int),
        })
        outcsv = group_dir / f"group_{name}_hemi-{hemi}.csv"
        df.to_csv(outcsv, index=False)

# ---------- GROUP plots ----------
import matplotlib.pyplot as plt

def plot_group(df, title, outf):
    x = df["mode_k"].values
    y = df["beta_mean"].values
    sem = df["beta_sem"].values
    q = df["q"].values
    sig = q < 0.05

    plt.figure(figsize=(7,4))
    plt.plot(x, y, lw=1.5)
    plt.fill_between(x, y-sem, y+sem, alpha=0.25)
    plt.scatter(x[sig], y[sig], s=25, color="red", zorder=5, label="q<0.05")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("Eigenmode index (k)")
    plt.ylabel("Boundary effect (β on energy)")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outf, dpi=150)
    plt.close()

for name in ["onset","offset"]:
    for hemi in ["L","R"]:
        df = pd.read_csv(group_dir / f"group_{name}_hemi-{hemi}.csv")
        plot_group(df, f"Group GLM — {name} boundaries (hemi {hemi})",
                   group_dir / f"group_{name}_hemi-{hemi}.png")

print(f"\n[DONE] Group summaries & plots in: {group_dir}")
print(" Files:")
for p in sorted(group_dir.glob("*")):
    print("  -", p)
PY