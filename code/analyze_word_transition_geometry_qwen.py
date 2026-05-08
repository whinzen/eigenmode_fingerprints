#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt


BASE = Path.home() / "eigenmode_fingerprints"

DEFAULT_EMB = BASE / "pang_out" / "token_embeddings_qwen" / "token_embeddings_qwen3_0p6b.pkl"
DEFAULT_OUT = BASE / "pang_out" / "word_transition_geometry_qwen3_0p6b"

WINDOW_RADIUS = 2
AR_ORDER = 3
SUBSPACE_WIN = 5
SUBSPACE_VAR_EXPLAINED = 0.90


def cosine_distance(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return np.nan
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return 1.0 - np.dot(a, b) / (na * nb)


def angle_between(u, v):
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        return np.nan
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return np.nan
    c = np.dot(u, v) / (nu * nv)
    c = np.clip(c, -1.0, 1.0)
    return np.arccos(c)


def normalize_embeddings(E):
    E = np.asarray(E, float)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return E / norms


def local_spectrum(X):
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[0] < 2:
        return np.array([np.nan])
    if not np.all(np.isfinite(X)):
        return np.array([np.nan])

    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(Xc, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.array([np.nan])

    eigs = (s ** 2) / max(1, X.shape[0] - 1)
    eigs = eigs[np.isfinite(eigs) & (eigs > 1e-12)]
    return eigs if len(eigs) else np.array([np.nan])


def geom_entropy_from_eigs(eigs):
    eigs = np.asarray(eigs, float)
    eigs = eigs[np.isfinite(eigs) & (eigs > 0)]
    if len(eigs) == 0:
        return np.nan
    p = eigs / eigs.sum()
    return -np.sum(p * np.log(p))


def participation_ratio_id(eigs):
    eigs = np.asarray(eigs, float)
    eigs = eigs[np.isfinite(eigs) & (eigs > 0)]
    if len(eigs) == 0:
        return np.nan
    den = np.sum(eigs ** 2)
    if den == 0:
        return np.nan
    return (eigs.sum() ** 2) / den


def fit_ar_weights(embs, run_ids, order=3, ridge=1e-6):
    ys = []
    xs = []

    N = len(embs)
    for t in range(order, N):
        if len(set(run_ids[t - order:t + 1])) != 1:
            continue

        y = np.asarray(embs[t], float)
        x = np.stack(
            [np.asarray(embs[t - j], float) for j in range(1, order + 1)],
            axis=1,
        )

        good = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
        if good.sum() == 0:
            continue

        ys.append(y[good])
        xs.append(x[good])

    if len(xs) == 0:
        return np.full(order, np.nan)

    Y = np.concatenate(ys, axis=0)
    X = np.concatenate(xs, axis=0)

    good_rows = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
    X = X[good_rows]
    Y = Y[good_rows]

    if len(Y) == 0:
        return np.full(order, np.nan)

    XtX = X.T @ X
    XtY = X.T @ Y

    try:
        w = np.linalg.solve(XtX + ridge * np.eye(order), XtY)
    except np.linalg.LinAlgError:
        return np.full(order, np.nan)

    return w


def ar_predict(embs, t, weights):
    order = len(weights)
    if t < order or not np.all(np.isfinite(weights)):
        return None

    pred = np.zeros_like(embs[t], dtype=float)

    for j, w in enumerate(weights, start=1):
        e = np.asarray(embs[t - j], float)
        if not np.all(np.isfinite(e)):
            return None
        pred += w * e

    return pred


def subspace_exit_error(embs, t, run_ids, subspace_win=5, var_explained=0.90):
    if t < subspace_win:
        return np.nan
    if len(set(run_ids[t - subspace_win:t + 1])) != 1:
        return np.nan

    Xhist = np.asarray(embs[t - subspace_win:t], float)
    x = np.asarray(embs[t], float)

    if not np.all(np.isfinite(Xhist)) or not np.all(np.isfinite(x)):
        return np.nan

    Xmean = Xhist.mean(axis=0)
    Xc = Xhist - Xmean
    xc = x - Xmean

    try:
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.nan

    eigs = s ** 2
    if eigs.sum() <= 0:
        return np.nan

    cum = np.cumsum(eigs) / eigs.sum()
    k = int(np.searchsorted(cum, var_explained) + 1)
    basis = Vt[:k].T

    proj = basis @ (basis.T @ xc)
    resid = xc - proj
    return np.linalg.norm(resid)


def summarize_group_difference(df, measure, flag_col):
    a = df.loc[df[flag_col] == 0, measure].dropna().values
    b = df.loc[df[flag_col] == 1, measure].dropna().values

    if len(a) < 2 or len(b) < 2:
        return {
            "measure": measure,
            "group0_mean": np.nan,
            "group0_sem": np.nan,
            "group1_mean": np.nan,
            "group1_sem": np.nan,
            "n_group0": len(a),
            "n_group1": len(b),
            "t": np.nan,
            "p": np.nan,
        }

    t, p = ttest_ind(b, a, equal_var=False, nan_policy="omit")

    return {
        "measure": measure,
        "group0_mean": float(np.mean(a)),
        "group0_sem": float(np.std(a, ddof=1) / np.sqrt(len(a))),
        "group1_mean": float(np.mean(b)),
        "group1_sem": float(np.std(b, ddof=1) / np.sqrt(len(b))),
        "n_group0": int(len(a)),
        "n_group1": int(len(b)),
        "t": float(t),
        "p": float(p),
    }


def save_hist_overlay(a, b, label_a, label_b, xlabel, title, outpath, bins=50):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(a, bins=bins, alpha=0.5, density=True, label=label_a)
    ax.hist(b, bins=bins, alpha=0.5, density=True, label=label_b)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def load_embedding_table(path):
    path = Path(path).expanduser()
    print(f"Loading embeddings: {path}")
    tok_df = pd.read_pickle(path)

    required = ["word", "snt_id", "token_id", "onset", "offset", "run_id", "embedding"]
    missing = [c for c in required if c not in tok_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in embedding table: {missing}")

    tok_df = tok_df.copy()

    # Important for Qwen replication:
    # the original BERT table contains one blank/space token; Qwen skipped it.
    # We therefore trust the supplied Qwen table and sort by actual timing.
    tok_df["word"] = tok_df["word"].astype(str).str.strip()
    tok_df["run_id"] = tok_df["run_id"].astype(int)
    tok_df["snt_id"] = tok_df["snt_id"].astype(int)
    tok_df["onset"] = tok_df["onset"].astype(float)
    tok_df["offset"] = tok_df["offset"].astype(float)

    tok_df = tok_df.sort_values(["run_id", "snt_id", "onset", "offset"]).reset_index(drop=True)

    # Create stable global index independent of original token_id convention.
    tok_df["token_id_original"] = tok_df["token_id"]
    tok_df["token_id"] = np.arange(1, len(tok_df) + 1)

    X = np.vstack(tok_df["embedding"].values)
    print("Loaded token dataframe:", tok_df.shape)
    print("Embedding matrix:", X.shape)
    print("NaNs:", int(np.isnan(X).sum()))

    if np.isnan(X).any():
        raise ValueError("Embedding matrix contains NaNs.")

    X = normalize_embeddings(X)
    tok_df["embedding"] = list(X)

    return tok_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-pkl", default=str(DEFAULT_EMB))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--window-radius", type=int, default=WINDOW_RADIUS)
    parser.add_argument("--ar-order", type=int, default=AR_ORDER)
    parser.add_argument("--subspace-win", type=int, default=SUBSPACE_WIN)
    parser.add_argument("--subspace-var", type=float, default=SUBSPACE_VAR_EXPLAINED)

    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_tok_copy = out_dir / "token_embeddings_contextual.pkl"
    out_state = out_dir / "word_state_geometry.csv"
    out_state_summary = out_dir / "word_state_summary.csv"
    out_trans = out_dir / "word_transition_geometry.csv"
    out_trans_summary = out_dir / "word_transition_summary.csv"
    out_boundary_state = out_dir / "boundary_state_geometry.csv"
    out_boundary_state_summary = out_dir / "boundary_state_summary.csv"

    tok_df = load_embedding_table(args.embeddings_pkl)
    tok_df.to_pickle(out_tok_copy)
    print(f"✅ wrote embedding copy: {out_tok_copy}")

    embs = np.vstack(tok_df["embedding"].values)
    run_ids = tok_df["run_id"].values
    N = len(tok_df)

    print(f"Fitting AR({args.ar_order}) weights...")
    ar_weights = fit_ar_weights(embs, run_ids, order=args.ar_order)
    print("AR weights:", ar_weights)

    # -------------------------
    # Local state geometry
    # -------------------------
    print("Computing local state geometry...")
    geom_H = np.full(N, np.nan)
    id_pr = np.full(N, np.nan)

    for i in tqdm(range(N), desc="Local geometry"):
        lo = max(0, i - args.window_radius)
        hi = min(N, i + args.window_radius + 1)
        Xw = embs[lo:hi]
        eigs = local_spectrum(Xw)
        geom_H[i] = geom_entropy_from_eigs(eigs)
        id_pr[i] = participation_ratio_id(eigs)

    tok_df["geom_entropy_state"] = geom_H
    tok_df["id_state"] = id_pr

    tok_df["sentence_initial_flag"] = 0
    tok_df.loc[1:, "sentence_initial_flag"] = (
        tok_df["snt_id"].values[1:] != tok_df["snt_id"].values[:-1]
    ).astype(int)

    state_summary_rows = []
    for measure in ["geom_entropy_state", "id_state"]:
        res = summarize_group_difference(tok_df, measure, "sentence_initial_flag")
        res["group0_label"] = "within_sentence_state"
        res["group1_label"] = "boundary_initial_state"
        state_summary_rows.append(res)

    state_summary_df = pd.DataFrame(state_summary_rows)

    tok_df.to_csv(out_state, index=False)
    state_summary_df.to_csv(out_state_summary, index=False)
    print(f"✅ wrote {out_state}")
    print(f"✅ wrote {out_state_summary}")

    # -------------------------
    # Transition metrics
    # -------------------------
    print("Computing transition metrics...")
    rows = []

    for i in tqdm(range(1, N), desc="Transitions"):
        prev = tok_df.iloc[i - 1]
        cur = tok_df.iloc[i]

        same_run = int(cur["run_id"] == prev["run_id"])
        boundary_flag = int(cur["snt_id"] != prev["snt_id"])

        shift = cosine_distance(cur["embedding"], prev["embedding"])
        dH = prev["geom_entropy_state"] - cur["geom_entropy_state"]
        dID = prev["id_state"] - cur["id_state"]

        pred_error_lin = np.nan
        if i >= 2:
            prev2 = tok_df.iloc[i - 2]
            if int(prev2["run_id"]) == int(prev["run_id"]) == int(cur["run_id"]):
                e_tm2 = np.asarray(prev2["embedding"], float)
                e_tm1 = np.asarray(prev["embedding"], float)
                e_t = np.asarray(cur["embedding"], float)
                e_pred = 2.0 * e_tm1 - e_tm2
                pred_error_lin = cosine_distance(e_pred, e_t)

        pred_error_ar = np.nan
        if (
            i >= args.ar_order
            and len(set(run_ids[i - args.ar_order:i + 1])) == 1
            and np.all(np.isfinite(ar_weights))
        ):
            e_pred_ar = ar_predict(embs, i, ar_weights)
            if e_pred_ar is not None:
                pred_error_ar = cosine_distance(e_pred_ar, embs[i])

        pred_error_subspace = subspace_exit_error(
            embs=embs,
            t=i,
            run_ids=run_ids,
            subspace_win=args.subspace_win,
            var_explained=args.subspace_var,
        )

        curvature = np.nan
        if i >= 2:
            prev2 = tok_df.iloc[i - 2]
            if int(prev2["run_id"]) == int(prev["run_id"]) == int(cur["run_id"]):
                e_tm2 = np.asarray(prev2["embedding"], float)
                e_tm1 = np.asarray(prev["embedding"], float)
                e_t = np.asarray(cur["embedding"], float)
                v1 = e_tm1 - e_tm2
                v2 = e_t - e_tm1
                curvature = angle_between(v1, v2)

        rows.append({
            "transition_idx": i,
            "prev_token_id": int(prev["token_id"]),
            "token_id": int(cur["token_id"]),
            "run_id": int(cur["run_id"]),
            "snt_prev": int(prev["snt_id"]),
            "snt_cur": int(cur["snt_id"]),
            "onset": float(cur["onset"]),
            "offset": float(cur["offset"]),
            "word_prev": str(prev["word"]),
            "word_cur": str(cur["word"]),
            "same_run": same_run,
            "boundary_flag": boundary_flag,
            "shift": shift,
            "entropy_prev": float(prev["geom_entropy_state"]),
            "entropy_cur": float(cur["geom_entropy_state"]),
            "entropy_change": dH,
            "id_prev": float(prev["id_state"]),
            "id_cur": float(cur["id_state"]),
            "id_change": dID,
            "pred_error_lin": pred_error_lin,
            "pred_error_ar": pred_error_ar,
            "pred_error_subspace": pred_error_subspace,
            "curvature": curvature,
        })

    trans_df = pd.DataFrame(rows)
    trans_df = trans_df[trans_df["same_run"] == 1].copy()

    transition_summary_rows = []
    for measure in [
        "shift",
        "entropy_change",
        "id_change",
        "pred_error_lin",
        "pred_error_ar",
        "pred_error_subspace",
        "curvature",
    ]:
        res = summarize_group_difference(trans_df, measure, "boundary_flag")
        res["group0_label"] = "within_sentence_transition"
        res["group1_label"] = "sentence_boundary_transition"
        transition_summary_rows.append(res)

    trans_summary_df = pd.DataFrame(transition_summary_rows)

    trans_df.to_csv(out_trans, index=False)
    trans_summary_df.to_csv(out_trans_summary, index=False)
    print(f"✅ wrote {out_trans}")
    print(f"✅ wrote {out_trans_summary}")

    # -------------------------
    # Boundary-state table
    # -------------------------
    boundary_state_df = tok_df[tok_df["sentence_initial_flag"] == 1].copy()
    boundary_state_df.to_csv(out_boundary_state, index=False)

    boundary_summary_rows = []
    for measure in ["geom_entropy_state", "id_state"]:
        res = summarize_group_difference(tok_df, measure, "sentence_initial_flag")
        res["group0_label"] = "non_initial_token"
        res["group1_label"] = "sentence_initial_token"
        boundary_summary_rows.append(res)

    boundary_summary_df = pd.DataFrame(boundary_summary_rows)
    boundary_summary_df.to_csv(out_boundary_state_summary, index=False)

    print(f"✅ wrote {out_boundary_state}")
    print(f"✅ wrote {out_boundary_state_summary}")

    # -------------------------
    # Simple diagnostic plots
    # -------------------------
    for measure in ["shift", "pred_error_ar", "pred_error_subspace", "curvature"]:
        save_hist_overlay(
            trans_df.loc[trans_df["boundary_flag"] == 0, measure].values,
            trans_df.loc[trans_df["boundary_flag"] == 1, measure].values,
            "within",
            "boundary",
            measure,
            f"Qwen transition metric: {measure}",
            fig_dir / f"hist_{measure}.png",
        )

    print("\nDone.")
    print("Output directory:", out_dir)
    print("Transition rows:", len(trans_df))
    print("Boundary transitions:", int(trans_df["boundary_flag"].sum()))


if __name__ == "__main__":
    main()