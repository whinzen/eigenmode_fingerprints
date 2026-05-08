import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.stats import ttest_ind, ttest_rel
import matplotlib.pyplot as plt

# =========================
# Paths / config
# =========================
BASE = Path.home() / "eigenmode_fingerprints"
CSV_PATH = BASE / "ds003643" / "annotation" / "EN" / "repunct" / "lppEN.csv"

OUT_DIR = BASE / "pang_out" / "word_transition_geometry"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_TRANS = OUT_DIR / "word_transition_geometry.csv"
OUT_TRANS_SUMMARY = OUT_DIR / "word_transition_summary.csv"

OUT_STATE = OUT_DIR / "word_state_geometry.csv"
OUT_STATE_SUMMARY = OUT_DIR / "word_state_summary.csv"

OUT_BOUNDARY_STATE = OUT_DIR / "boundary_state_geometry.csv"
OUT_BOUNDARY_STATE_SUMMARY = OUT_DIR / "boundary_state_summary.csv"

OUT_TOK_PKL = OUT_DIR / "token_embeddings_contextual.pkl"

MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
WINDOW_RADIUS = 2
STATE_WIN = 4
AR_ORDER = 3
SUBSPACE_WIN = 5
SUBSPACE_VAR_EXPLAINED = 0.90
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Geometry helpers
# =========================
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
    if len(eigs) == 0:
        return np.array([np.nan])
    return eigs

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

# =========================
# Prediction helpers
# =========================
def fit_ar_weights(embs, run_ids, order=3, ridge=1e-6):
    ys = []
    xs = []

    N = len(embs)
    for t in range(order, N):
        if len(set(run_ids[t - order:t + 1])) != 1:
            continue

        y = np.asarray(embs[t], float)
        x = np.stack([np.asarray(embs[t - j], float) for j in range(1, order + 1)], axis=1)

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

    hist = []
    for j in range(1, order + 1):
        e = np.asarray(embs[t - j], float)
        if not np.all(np.isfinite(e)):
            return None
        hist.append(e)

    pred = np.zeros_like(embs[t], dtype=float)
    for w, e in zip(weights, hist):
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

    Xc = Xhist - Xhist.mean(axis=0, keepdims=True)
    xc = x - Xhist.mean(axis=0)

    try:
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.nan

    eigs = s ** 2
    if np.sum(eigs) <= 0:
        return np.nan

    cum = np.cumsum(eigs) / np.sum(eigs)
    k = int(np.searchsorted(cum, var_explained) + 1)
    basis = Vt[:k].T

    proj = basis @ (basis.T @ xc)
    resid = xc - proj
    return np.linalg.norm(resid)

# =========================
# Stats / plotting helpers
# =========================
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
        "group0_mean": np.mean(a),
        "group0_sem": np.std(a, ddof=1) / np.sqrt(len(a)),
        "group1_mean": np.mean(b),
        "group1_sem": np.std(b, ddof=1) / np.sqrt(len(b)),
        "n_group0": len(a),
        "n_group1": len(b),
        "t": t,
        "p": p,
    }

def save_barplot_with_sem(labels, means, sems, ylabel, title, outpath):
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, means, yerr=sems, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"✅ wrote {outpath}")

def save_hist_overlay(a, b, label_a, label_b, xlabel, title, outpath, bins=50):
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
    print(f"✅ wrote {outpath}")

def save_scatter(x, y, cflag, xlabel, ylabel, title, outpath, max_points=5000):
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(cflag)
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]
    cflag = np.asarray(cflag)[mask]

    if len(x) > max_points:
        idx = np.random.default_rng(0).choice(len(x), size=max_points, replace=False)
        x = x[idx]
        y = y[idx]
        cflag = cflag[idx]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x[cflag == 0], y[cflag == 0], s=8, alpha=0.35, label="Within")
    ax.scatter(x[cflag == 1], y[cflag == 1], s=8, alpha=0.35, label="Boundary")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"✅ wrote {outpath}")

# =========================
# Embedding build / load
# =========================
def encode_sentence_words_batch(word_lists, tokenizer, model, batch_size=16, device="cpu"):
    all_outputs = []

    for start in tqdm(range(0, len(word_lists), batch_size), desc="Encoding batches"):
        batch_words = word_lists[start:start + batch_size]

        enc = tokenizer(
            batch_words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        batch_word_ids = [enc.word_ids(batch_index=i) for i in range(len(batch_words))]
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            if device == "cuda":
                with torch.cuda.amp.autocast():
                    out = model(**enc)
            else:
                out = model(**enc)

        hidden = out.last_hidden_state.cpu().numpy()

        for b_idx, words in enumerate(batch_words):
            word_ids = batch_word_ids[b_idx]
            sent_hidden = hidden[b_idx]

            word_embs = []
            for i in range(len(words)):
                idx = [j for j, wid in enumerate(word_ids) if wid == i]
                if len(idx) == 0:
                    word_embs.append(np.full(sent_hidden.shape[1], np.nan))
                else:
                    word_embs.append(sent_hidden[idx].mean(axis=0))
            all_outputs.append(np.vstack(word_embs))

    return all_outputs

def build_or_load_token_embeddings(df, out_tok_pkl, model_name, batch_size, device):
    if out_tok_pkl.exists():
        print(f"✅ Loading cached token embeddings from {out_tok_pkl}")
        return pd.read_pickle(out_tok_pkl)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    groups = []
    word_lists = []

    for _, g in df.groupby("snt_id", sort=True):
        g = g.sort_values("token_id").copy()
        groups.append(g)
        word_lists.append(g["word"].astype(str).tolist())

    print("Encoding contextual word embeddings in batches...")
    all_embs = encode_sentence_words_batch(word_lists, tokenizer, model, batch_size=batch_size, device=device)

    all_rows = []
    for g, word_embs in zip(groups, all_embs):
        if len(word_embs) != len(g):
            raise RuntimeError("Embedding/word count mismatch.")
        g = g.copy()
        g["embedding"] = list(word_embs)
        all_rows.append(g)

    tok_df = pd.concat(all_rows, ignore_index=True)
    print(f"✅ Saving token embedding cache to {out_tok_pkl}")
    tok_df.to_pickle(out_tok_pkl)
    return tok_df

# =========================
# Main
# =========================
def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    required = ["word", "snt_id", "token_id", "onset", "offset", "run_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["run_id"] = df["run_id"].astype(int)
    df["snt_id"] = df["snt_id"].astype(int)
    df["token_id"] = df["token_id"].astype(int)
    df = df.sort_values(["run_id", "snt_id", "token_id"]).reset_index(drop=True)

    tok_df = build_or_load_token_embeddings(
        df=df,
        out_tok_pkl=OUT_TOK_PKL,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )
    print(f"Loaded token dataframe with {len(tok_df)} rows")

    # normalize embeddings
    norm_embs = normalize_embeddings(np.vstack(tok_df["embedding"].values))
    tok_df["embedding"] = list(norm_embs)
    embs = np.vstack(tok_df["embedding"].values)
    run_ids = tok_df["run_id"].values
    N = len(tok_df)
    print("Embeddings normalized")

    # sentence lengths for adaptive windows
    sent_lengths = tok_df.groupby("snt_id").size().to_dict()

    print(f"Fitting AR({AR_ORDER}) weights...")
    ar_weights = fit_ar_weights(embs, run_ids, order=AR_ORDER)
    print("AR weights:", ar_weights)

    # =========================
    # Generic state metrics
    # =========================
    print("Computing generic state metrics...")
    geom_H = np.full(N, np.nan)
    id_pr = np.full(N, np.nan)

    for i in tqdm(range(N), desc="Local geometry"):
        lo = max(0, i - WINDOW_RADIUS)
        hi = min(N, i + WINDOW_RADIUS + 1)
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

    # =========================
    # Transition metrics
    # =========================
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
        if i >= AR_ORDER and len(set(run_ids[i - AR_ORDER:i + 1])) == 1 and np.all(np.isfinite(ar_weights)):
            e_pred_ar = ar_predict(embs, i, ar_weights)
            if e_pred_ar is not None:
                pred_error_ar = cosine_distance(e_pred_ar, embs[i])

        pred_error_subspace = subspace_exit_error(
            embs=embs,
            t=i,
            run_ids=run_ids,
            subspace_win=SUBSPACE_WIN,
            var_explained=SUBSPACE_VAR_EXPLAINED
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
        res["group0_label"] = "within_transition"
        res["group1_label"] = "boundary_transition"
        transition_summary_rows.append(res)
    transition_summary_df = pd.DataFrame(transition_summary_rows)

    # =========================
    # Pre/post boundary states
    # =========================
    print("Computing pre/post boundary states...")
    boundary_rows = []

    for i in tqdm(range(1, N), desc="Boundary states"):
        prev = tok_df.iloc[i - 1]
        cur = tok_df.iloc[i]

        if int(cur["run_id"]) != int(prev["run_id"]):
            continue
        if int(cur["snt_id"]) == int(prev["snt_id"]):
            continue

        prev_sent_len = sent_lengths.get(int(prev["snt_id"]), 0)
        cur_sent_len = sent_lengths.get(int(cur["snt_id"]), 0)

        pre_w = min(STATE_WIN, prev_sent_len)
        post_w = min(STATE_WIN, cur_sent_len)

        if pre_w < 2 or post_w < 2:
            continue

        pre_start = i - pre_w
        pre_end = i
        post_start = i
        post_end = i + post_w

        if pre_start < 0 or post_end > N:
            continue

        pre_df = tok_df.iloc[pre_start:pre_end]
        post_df = tok_df.iloc[post_start:post_end]

        if pre_df["run_id"].nunique() != 1 or post_df["run_id"].nunique() != 1:
            continue
        if int(pre_df["run_id"].iloc[0]) != int(post_df["run_id"].iloc[0]):
            continue

        if pre_df["snt_id"].nunique() != 1 or post_df["snt_id"].nunique() != 1:
            continue
        if int(pre_df["snt_id"].iloc[0]) != int(prev["snt_id"]):
            continue
        if int(post_df["snt_id"].iloc[0]) != int(cur["snt_id"]):
            continue

        Xpre = np.vstack(pre_df["embedding"].values)
        Xpost = np.vstack(post_df["embedding"].values)

        eigs_pre = local_spectrum(Xpre)
        eigs_post = local_spectrum(Xpost)

        H_pre = geom_entropy_from_eigs(eigs_pre)
        H_post = geom_entropy_from_eigs(eigs_post)
        ID_pre = participation_ratio_id(eigs_pre)
        ID_post = participation_ratio_id(eigs_post)

        boundary_rows.append({
            "boundary_idx": i,
            "run_id": int(cur["run_id"]),
            "prev_sentence_id": int(prev["snt_id"]),
            "cur_sentence_id": int(cur["snt_id"]),
            "boundary_onset": float(cur["onset"]),
            "pre_window_size": pre_w,
            "post_window_size": post_w,
            "pre_entropy": H_pre,
            "post_entropy": H_post,
            "pre_id": ID_pre,
            "post_id": ID_post,
            "entropy_post_minus_pre": H_post - H_pre,
            "id_post_minus_pre": ID_post - ID_pre,
        })

    boundary_state_df = pd.DataFrame(boundary_rows)

    boundary_state_summary_rows = []
    for pre_col, post_col, name in [
        ("pre_entropy", "post_entropy", "geom_entropy_prepost"),
        ("pre_id", "post_id", "id_prepost"),
    ]:
        good = boundary_state_df[[pre_col, post_col]].dropna()
        if len(good) >= 2:
            t, p = ttest_rel(good[post_col].values, good[pre_col].values)
            boundary_state_summary_rows.append({
                "measure": name,
                "pre_mean": good[pre_col].mean(),
                "pre_sem": good[pre_col].std(ddof=1) / np.sqrt(len(good)),
                "post_mean": good[post_col].mean(),
                "post_sem": good[post_col].std(ddof=1) / np.sqrt(len(good)),
                "post_minus_pre_mean": (good[post_col] - good[pre_col]).mean(),
                "N": len(good),
                "t": t,
                "p": p,
            })
        else:
            boundary_state_summary_rows.append({
                "measure": name,
                "pre_mean": np.nan,
                "pre_sem": np.nan,
                "post_mean": np.nan,
                "post_sem": np.nan,
                "post_minus_pre_mean": np.nan,
                "N": len(good),
                "t": np.nan,
                "p": np.nan,
            })

    boundary_state_summary_df = pd.DataFrame(boundary_state_summary_rows)

    # rename summary columns
    for sdf in [state_summary_df, transition_summary_df]:
        sdf.rename(columns={
            "group0_mean": "within_mean",
            "group0_sem": "within_sem",
            "group1_mean": "boundary_mean",
            "group1_sem": "boundary_sem",
            "n_group0": "n_within",
            "n_group1": "n_boundary",
        }, inplace=True)

    # =========================
    # Save tables
    # =========================
    print("Saving tables...")
    tok_df.to_csv(OUT_STATE, index=False)
    trans_df.to_csv(OUT_TRANS, index=False)
    boundary_state_df.to_csv(OUT_BOUNDARY_STATE, index=False)
    state_summary_df.to_csv(OUT_STATE_SUMMARY, index=False)
    transition_summary_df.to_csv(OUT_TRANS_SUMMARY, index=False)
    boundary_state_summary_df.to_csv(OUT_BOUNDARY_STATE_SUMMARY, index=False)

    print("\n=== Generic state summary ===")
    print(state_summary_df)
    print("\n=== Transition summary ===")
    print(transition_summary_df)
    print("\n=== Boundary pre/post state summary ===")
    print(boundary_state_summary_df)

    # =========================
    # Plots
    # =========================
    print("Saving plots...")

    for _, row in state_summary_df.iterrows():
        save_barplot_with_sem(
            labels=["Within", "Boundary-initial"],
            means=[row["within_mean"], row["boundary_mean"]],
            sems=[row["within_sem"], row["boundary_sem"]],
            ylabel=row["measure"],
            title=f"State comparison: {row['measure']}",
            outpath=FIG_DIR / f"state_{row['measure']}_bar.png"
        )

    for _, row in transition_summary_df.iterrows():
        save_barplot_with_sem(
            labels=["Within", "Boundary"],
            means=[row["within_mean"], row["boundary_mean"]],
            sems=[row["within_sem"], row["boundary_sem"]],
            ylabel=row["measure"],
            title=f"Transition comparison: {row['measure']}",
            outpath=FIG_DIR / f"transition_{row['measure']}_bar.png"
        )

    for _, row in boundary_state_summary_df.iterrows():
        save_barplot_with_sem(
            labels=["Pre", "Post"],
            means=[row["pre_mean"], row["post_mean"]],
            sems=[row["pre_sem"], row["post_sem"]],
            ylabel=row["measure"],
            title=f"Boundary pre/post: {row['measure']}",
            outpath=FIG_DIR / f"boundary_prepost_{row['measure']}_bar.png"
        )

    for measure in [
        "shift",
        "pred_error_lin",
        "pred_error_ar",
        "pred_error_subspace",
        "curvature",
        "entropy_change",
        "id_change",
    ]:
        within = trans_df.loc[trans_df["boundary_flag"] == 0, measure].dropna().values
        boundary = trans_df.loc[trans_df["boundary_flag"] == 1, measure].dropna().values
        if len(within) > 0 and len(boundary) > 0:
            save_hist_overlay(
                within, boundary,
                "Within", "Boundary",
                xlabel=measure,
                title=f"Transition distributions: {measure}",
                outpath=FIG_DIR / f"transition_{measure}_hist.png",
                bins=60
            )

    save_scatter(
        x=trans_df["shift"].values,
        y=trans_df["pred_error_lin"].values,
        cflag=trans_df["boundary_flag"].values,
        xlabel="Shift",
        ylabel="Linear prediction error",
        title="Shift vs prediction error (linear)",
        outpath=FIG_DIR / "shift_vs_pred_error_lin_scatter.png"
    )

    save_scatter(
        x=trans_df["shift"].values,
        y=trans_df["pred_error_ar"].values,
        cflag=trans_df["boundary_flag"].values,
        xlabel="Shift",
        ylabel="AR prediction error",
        title="Shift vs prediction error (AR)",
        outpath=FIG_DIR / "shift_vs_pred_error_ar_scatter.png"
    )

    save_scatter(
        x=trans_df["shift"].values,
        y=trans_df["pred_error_subspace"].values,
        cflag=trans_df["boundary_flag"].values,
        xlabel="Shift",
        ylabel="Subspace-exit error",
        title="Shift vs subspace-exit error",
        outpath=FIG_DIR / "shift_vs_pred_error_subspace_scatter.png"
    )

    print("✅ Done")

if __name__ == "__main__":
    main()