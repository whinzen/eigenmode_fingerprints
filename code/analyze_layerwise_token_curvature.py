import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# =========================
# Paths / config
# =========================
BASE = Path.home() / "eigenmode_fingerprints"
CSV_PATH = BASE / "ds003643" / "annotation" / "EN" / "repunct" / "lppEN.csv"

OUT_DIR = BASE / "pang_out" / "layerwise_token_curvature"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_TOKEN = OUT_DIR / "token_layerwise_curvature.csv"
OUT_SUMMARY = OUT_DIR / "layerwise_curvature_summary.csv"
OUT_LAYER_PROFILE = OUT_DIR / "layerwise_curvature_by_layer.csv"
OUT_SENT_PKL = OUT_DIR / "token_layer_hiddenstates.pkl"

MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Geometry helpers
# =========================
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

def sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    return x.std(ddof=1) / np.sqrt(len(x))

# =========================
# Plot helpers
# =========================
def save_barplot_with_sem(labels, means, sems, ylabel, title, outpath):
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(x, means, yerr=sems, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"✅ wrote {outpath}")

def save_profile_plot(df, outpath):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(df["layer_idx"], df["within_mean"], linewidth=2, label="Within-sentence")
    ax.fill_between(
        df["layer_idx"],
        df["within_mean"] - df["within_sem"],
        df["within_mean"] + df["within_sem"],
        alpha=0.2
    )

    ax.plot(df["layer_idx"], df["boundary_mean"], linewidth=2, label="Sentence-initial")
    ax.fill_between(
        df["layer_idx"],
        df["boundary_mean"] - df["boundary_sem"],
        df["boundary_mean"] + df["boundary_sem"],
        alpha=0.2
    )

    ax.set_xlabel("Layer transition index")
    ax.set_ylabel("Mean curvature (radians)")
    ax.set_title("Layerwise token curvature across Transformer depth")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"✅ wrote {outpath}")

# =========================
# Hidden-state extraction
# =========================
def extract_token_layer_states_batch(word_lists, tokenizer, model, batch_size=8, device="cpu"):
    """
    Returns one entry per sentence.
    Each entry is a list of length n_words.
    Each word is represented by an array of shape [n_layers+1, hidden_dim],
    because BERT hidden_states includes embeddings + all transformer layers.
    """
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
            out = model(**enc, output_hidden_states=True)

        # tuple length = n_layers+1, each [B, T, D]
        hidden_states = [h.cpu().numpy() for h in out.hidden_states]
        n_layer_states = len(hidden_states)

        for b_idx, words in enumerate(batch_words):
            word_ids = batch_word_ids[b_idx]
            token_word_states = []

            for i_word in range(len(words)):
                sub_idx = [j for j, wid in enumerate(word_ids) if wid == i_word]

                if len(sub_idx) == 0:
                    # no subwords found (rare edge case)
                    D = hidden_states[0].shape[-1]
                    token_word_states.append(np.full((n_layer_states, D), np.nan))
                    continue

                # For each layer, average subword states belonging to this word
                per_layer = []
                for l in range(n_layer_states):
                    layer_repr = hidden_states[l][b_idx]           # [T, D]
                    per_layer.append(layer_repr[sub_idx].mean(axis=0))
                token_word_states.append(np.vstack(per_layer))     # [L+1, D]

            all_outputs.append(token_word_states)

    return all_outputs

def build_or_load_token_layer_states(df, out_pkl, model_name, batch_size, device):
    if out_pkl.exists():
        print(f"✅ Loading cached token layer states from {out_pkl}")
        return pd.read_pickle(out_pkl)

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

    print("Encoding token hidden states across layers...")
    all_states = extract_token_layer_states_batch(
        word_lists, tokenizer, model,
        batch_size=batch_size, device=device
    )

    all_rows = []
    for g, token_states in zip(groups, all_states):
        if len(token_states) != len(g):
            raise RuntimeError("Layer-state/token count mismatch.")
        g = g.copy()
        g["layer_states"] = list(token_states)
        all_rows.append(g)

    tok_df = pd.concat(all_rows, ignore_index=True)
    print(f"✅ Saving token layer-state cache to {out_pkl}")
    tok_df.to_pickle(out_pkl)
    return tok_df

# =========================
# Curvature computation
# =========================
def compute_layerwise_curvature_for_token(layer_states):
    """
    layer_states: [n_layer_states, D]
    Curvature is computed across successive layerwise step vectors:
        v_l   = h_{l+1} - h_l
        curv_l = angle(v_l, v_{l+1})
    Returns:
        curv_by_layer: [n_layer_states - 2]
        curv_mean: scalar mean across valid layer transitions
    """
    H = np.asarray(layer_states, float)
    if H.ndim != 2 or H.shape[0] < 3:
        return np.array([np.nan]), np.nan

    steps = H[1:] - H[:-1]   # [n_layer_states-1, D]
    curvs = []
    for l in range(len(steps) - 1):
        curvs.append(angle_between(steps[l], steps[l + 1]))
    curvs = np.asarray(curvs, float)

    good = np.isfinite(curvs)
    curv_mean = np.nan if good.sum() == 0 else np.nanmean(curvs[good])

    return curvs, curv_mean

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

    tok_df = build_or_load_token_layer_states(
        df=df,
        out_pkl=OUT_SENT_PKL,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )
    print(f"Loaded token dataframe with {len(tok_df)} rows")

    # sentence-initial flag
    tok_df["sentence_initial_flag"] = 0
    tok_df.loc[1:, "sentence_initial_flag"] = (
        tok_df["snt_id"].values[1:] != tok_df["snt_id"].values[:-1]
    ).astype(int)

    print("Computing layerwise token curvature...")
    all_curvatures = []
    curv_means = []

    for layer_states in tqdm(tok_df["layer_states"].values, desc="Curvature"):
        curv_by_layer, curv_mean = compute_layerwise_curvature_for_token(layer_states)
        all_curvatures.append(curv_by_layer)
        curv_means.append(curv_mean)

    tok_df["layerwise_curvature"] = all_curvatures
    tok_df["layerwise_curvature_mean"] = curv_means

    # =========================
    # Summary comparison
    # =========================
    within = tok_df.loc[tok_df["sentence_initial_flag"] == 0, "layerwise_curvature_mean"].dropna().values
    boundary = tok_df.loc[tok_df["sentence_initial_flag"] == 1, "layerwise_curvature_mean"].dropna().values

    if len(within) >= 2 and len(boundary) >= 2:
        t, p = ttest_ind(boundary, within, equal_var=False, nan_policy="omit")
    else:
        t, p = np.nan, np.nan

    summary_df = pd.DataFrame([{
        "measure": "layerwise_token_curvature",
        "within_mean": np.nanmean(within) if len(within) else np.nan,
        "within_sem": sem(within),
        "boundary_mean": np.nanmean(boundary) if len(boundary) else np.nan,
        "boundary_sem": sem(boundary),
        "n_within": len(within),
        "n_boundary": len(boundary),
        "t": t,
        "p": p,
        "group0_label": "within_sentence_token",
        "group1_label": "sentence_initial_token",
    }])

    # =========================
    # By-layer profile
    # =========================
    # stack per-token curvature vectors
    max_len = max(len(c) for c in all_curvatures if isinstance(c, np.ndarray))
    rows = []

    for _, row in tok_df.iterrows():
        curv = np.asarray(row["layerwise_curvature"], float)
        for layer_idx, val in enumerate(curv):
            rows.append({
                "sentence_initial_flag": int(row["sentence_initial_flag"]),
                "layer_idx": int(layer_idx),
                "curvature": float(val),
            })

    layer_df = pd.DataFrame(rows)

    profile_rows = []
    for l, g in layer_df.groupby("layer_idx"):
        a = g.loc[g["sentence_initial_flag"] == 0, "curvature"].dropna().values
        b = g.loc[g["sentence_initial_flag"] == 1, "curvature"].dropna().values

        if len(a) >= 2 and len(b) >= 2:
            t_l, p_l = ttest_ind(b, a, equal_var=False, nan_policy="omit")
        else:
            t_l, p_l = np.nan, np.nan

        profile_rows.append({
            "layer_idx": int(l),
            "within_mean": np.nanmean(a) if len(a) else np.nan,
            "within_sem": sem(a),
            "boundary_mean": np.nanmean(b) if len(b) else np.nan,
            "boundary_sem": sem(b),
            "n_within": len(a),
            "n_boundary": len(b),
            "t": t_l,
            "p": p_l,
        })

    profile_df = pd.DataFrame(profile_rows).sort_values("layer_idx")

    # =========================
    # Save
    # =========================
    save_df = tok_df.drop(columns=["layer_states"])
    save_df.to_csv(OUT_TOKEN, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    profile_df.to_csv(OUT_LAYER_PROFILE, index=False)

    print("\n=== Layerwise curvature summary ===")
    print(summary_df)

    print("\n=== Layerwise curvature by layer ===")
    print(profile_df.head(15))

    # =========================
    # Plots
    # =========================
    save_barplot_with_sem(
        labels=["Within", "Sentence-initial"],
        means=[summary_df.loc[0, "within_mean"], summary_df.loc[0, "boundary_mean"]],
        sems=[summary_df.loc[0, "within_sem"], summary_df.loc[0, "boundary_sem"]],
        ylabel="Mean layerwise curvature (radians)",
        title="Layerwise token curvature",
        outpath=FIG_DIR / "layerwise_token_curvature_bar.png"
    )

    save_profile_plot(
        profile_df,
        outpath=FIG_DIR / "layerwise_token_curvature_by_layer.png"
    )

    print("✅ Done")

if __name__ == "__main__":
    main()