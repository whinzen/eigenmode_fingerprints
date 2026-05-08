import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.stats import gamma

BASE = Path.home() / "eigenmode_fingerprints"
CSV_PATH = BASE / "ds003643/annotation/EN/repunct/lppEN.csv"
OUT_DIR = BASE / "pang_out/regressors/sentence_shift_hrf_per_subject"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TR = 2.0
MODEL_NAME = "bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def spm_hrf(tr, length=32.0):
    t = np.arange(0, length, tr)
    peak = gamma.pdf(t, 6)
    undershoot = gamma.pdf(t, 16)
    hrf = peak - 0.5 * undershoot
    return hrf / hrf.max()

HRF = spm_hrf(TR)

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

df = pd.read_csv(CSV_PATH)
df = df.sort_values(["run_id", "snt_id", "token_id"])

sentences = df.groupby("snt_id")["word"].apply(list)

sentence_ids = list(sentences.index)
sentence_lists = list(sentences.values)

BATCH = 16  # try 32 if GPU memory allows

embs = {}

for i in range(0, len(sentence_lists), BATCH):
    batch = sentence_lists[i:i+BATCH]

    enc = tok(
        batch,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc).last_hidden_state  # shape: [B, T, D]

    sent_embs = out.mean(dim=1).cpu().numpy()  # shape: [B, D]

    for j, sid in enumerate(sentence_ids[i:i+BATCH]):
        embs[sid] = sent_embs[j]

df["boundary"] = (df["snt_id"] != df["snt_id"].shift(1)).astype(int)
df.loc[0, "boundary"] = 1

for run, g in df.groupby("run_id"):
    T = int(np.ceil(g["offset"].max() / TR))
    x = np.zeros(T)

    for i in range(1, len(g)):
        if g.iloc[i]["boundary"] == 1:
            s_prev = g.iloc[i-1]["snt_id"]
            s_cur = g.iloc[i]["snt_id"]

            shift = cosine_distance(embs[s_prev], embs[s_cur])

            idx = int(round(g.iloc[i]["onset"] / TR))
            if idx < T:
                x[idx] = shift

    x_hrf = np.convolve(x, HRF)[:T]
    np.save(OUT_DIR / f"run-{run:02d}.npy", x_hrf)

print("✅ Sentence shift regressors built")