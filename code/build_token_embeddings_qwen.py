#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE = Path.home() / "eigenmode_fingerprints"
CSV_PATH = BASE / "ds003643" / "annotation" / "EN" / "repunct" / "lppEN.csv"
OUT_DIR = BASE / "pang_out" / "token_embeddings_qwen"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(f"Could not find any of {candidates}. Found: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--layer", type=int, default=-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--out-name", default="token_embeddings_qwen3_0p6b.pkl")
    args = ap.parse_args()

    df = pd.read_csv(CSV_PATH).copy()

    word_col = find_col(df, ["word", "token"])
    sent_col = find_col(df, ["snt_id", "sent_id", "sentence_id"])
    onset_col = find_col(df, ["onset", "onset_sec", "word_onset", "start"])
    offset_col = find_col(df, ["offset", "offset_sec", "word_offset", "end"])
    run_col = find_col(df, ["run_id", "run"])

    df[word_col] = df[word_col].astype(str).str.strip()
    df = df.sort_values([run_col, sent_col, onset_col]).reset_index(drop=True)

    print("Loading tokenizer/model:", args.model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    print("Device:", device)
    print("Layer:", args.layer)
    print("Batch size:", args.batch_size)

    sent_items = []
    for (run_id, snt_id), g in df.groupby([run_col, sent_col], sort=True):
        g = g.sort_values(onset_col).reset_index(drop=True)
        words = g[word_col].tolist()
        sent_items.append((run_id, snt_id, g, words))

    rows = []
    token_id = 1
    n_missing = 0

    for start in range(0, len(sent_items), args.batch_size):
        batch = sent_items[start:start + args.batch_size]
        batch_words = [item[3] for item in batch]

        enc = tokenizer(
            batch_words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=True,
        )

        # Save word_id maps before moving tensors
        word_id_maps = [enc.word_ids(batch_index=i) for i in range(len(batch_words))]
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(
                **enc,
                output_hidden_states=True,
                use_cache=False,
            )

        H = out.hidden_states[args.layer].detach().float().cpu().numpy()
        # H shape: [batch, subword_tokens, hidden_dim]

        for bi, (run_id, snt_id, g, words) in enumerate(batch):
            word_ids = word_id_maps[bi]

            for local_i, row in g.iterrows():
                sub_idx = [j for j, wid in enumerate(word_ids) if wid == local_i]

                if len(sub_idx) == 0:
                    n_missing += 1
                    continue

                emb = H[bi, sub_idx, :].mean(axis=0).astype(np.float32)

                rows.append({
                    "word": str(row[word_col]),
                    "snt_id": int(row[sent_col]),
                    "token_id": token_id,
                    "onset": float(row[onset_col]),
                    "offset": float(row[offset_col]),
                    "run_id": int(row[run_col]),
                    "embedding": emb,
                })
                token_id += 1

        if len(rows) % 1000 < 100:
            print(f"embedded rows: {len(rows)} / {len(df)}")

    out_df = pd.DataFrame(rows)
    out_path = OUT_DIR / args.out_name
    out_df.to_pickle(out_path)

    X = np.vstack(out_df["embedding"].values)

    print("\nDone.")
    print("Output:", out_path)
    print("DataFrame shape:", out_df.shape)
    print("Embedding matrix:", X.shape)
    print("Missing words:", n_missing)
    print("NaNs:", np.isnan(X).sum())


if __name__ == "__main__":
    main()