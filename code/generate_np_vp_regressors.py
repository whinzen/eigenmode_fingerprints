import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tree import Tree

# 📁 Input paths
CSV_TREE = "../ds003643/annotation/EN/lppEN_tree.csv"
CSV_WORDS = "../ds003643/annotation/EN/repunct/lppEN.csv"
CSV_TRS = "../ds003643/annotation/EN/num_trs_per_run.csv"

# 💾 Output paths
OUTDIR_NP = "../pang_out/regressors/boundary_np"
OUTDIR_VP = "../pang_out/regressors/boundary_vp"
os.makedirs(OUTDIR_NP, exist_ok=True)
os.makedirs(OUTDIR_VP, exist_ok=True)

# 🗺️ Map annotation run_id (1-9) to fMRI BOLD run_id (15-23)
RUN_ID_MAP = {i + 1: i + 15 for i in range(9)}

print("\U0001F4C2 Loading input files...")
df_tree = pd.read_csv(CSV_TREE, header=None, names=["tree"])
df_words = pd.read_csv(CSV_WORDS)
df_trs = pd.read_csv(CSV_TRS)
df_tree["snt_id"] = df_tree.index + 1

# Merge to get run_id per sentence
df_words_sent = df_words.drop_duplicates("snt_id")[["snt_id", "run_id"]]
df_tree = df_tree.merge(df_words_sent, on="snt_id", how="left")

# Helper: extract terminal positions of NP and VP

def extract_np_vp_ends(tree_str):
    try:
        tree = Tree.fromstring(tree_str)
    except Exception:
        return [], []

    np_ends, vp_ends = [], []

    def recurse(t, pos=0):
        if isinstance(t, str):
            return 1  # terminal

        n_tokens = 0
        for child in t:
            n = recurse(child, pos + n_tokens)
            n_tokens += n

        label = t.label() if hasattr(t, 'label') else ""
        if label == "NP":
            np_ends.append(pos + n_tokens - 1)
        elif label == "VP":
            vp_ends.append(pos + n_tokens - 1)

        return n_tokens

    recurse(tree)
    return np_ends, vp_ends

print("\U0001F332 Parsing constituency trees and aligning boundaries...")
df_tree["np_ends"], df_tree["vp_ends"] = zip(*df_tree["tree"].map(extract_np_vp_ends))

# Merge NP/VP ends with full token list
df_full = df_words.merge(df_tree[["snt_id", "np_ends", "vp_ends"]], on="snt_id", how="left")

# Loop over actual BOLD run IDs 15–23
print("\U0001F4BE Saving per-run binary regressors...")
for bold_run_id in range(15, 24):
    # Match annotation run ID
    run_id_old = bold_run_id - 14  # 15 -> 1, ..., 23 -> 9
    df_run = df_full[df_full.run_id == run_id_old].copy()

    if df_run.empty:
        print(f"⚠️  No data for run {bold_run_id}, skipping...")
        continue

    n_trs_row = df_trs[df_trs.run_id == bold_run_id]
    if n_trs_row.empty:
        print(f"⚠️  No TR info for run {bold_run_id}, skipping...")
        continue
    n_trs = int(n_trs_row["num_trs"].values[0])

    # Initialize empty regressor
    reg_np = np.zeros(n_trs)
    reg_vp = np.zeros(n_trs)

    # Fill binary regressors at TR bins corresponding to NP/VP ends
    for snt_id, group in df_run.groupby("snt_id"):
        try:
            token_ids = list(group["token_id"])
            onsets = list(group["onset"])
            if not onsets:
                continue
            for idx in group["np_ends"].iloc[0]:
                if idx < len(onsets):
                    tr_idx = int(onsets[idx] // 2.0)
                    if tr_idx < n_trs:
                        reg_np[tr_idx] = 1
            for idx in group["vp_ends"].iloc[0]:
                if idx < len(onsets):
                    tr_idx = int(onsets[idx] // 2.0)
                    if tr_idx < n_trs:
                        reg_vp[tr_idx] = 1
        except Exception:
            continue

    # Save as .npy
    np.save(os.path.join(OUTDIR_NP, f"run-{bold_run_id}.npy"), reg_np)
    np.save(os.path.join(OUTDIR_VP, f"run-{bold_run_id}.npy"), reg_vp)
    print(f"  → Run {bold_run_id}: {int(reg_np.sum())} NP, {int(reg_vp.sum())} VP events")

print("✅ Done.")