#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE = Path.home() / "eigenmode_fingerprints"
CSV_PATH = BASE / "ds003643" / "annotation" / "EN" / "repunct" / "lppEN.csv"

OUT_DIR = BASE / "pang_out" / "token_embeddings_qwen3_4b_proportional"


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"Could not find any of {candidates}. Found: {list(df.columns)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--layer", type=int, default=-5)
    ap.add_argument(
        "--out-name",
        default="token_embeddings_qwen3_4b_continuous_layer_m5.pkl",
    )
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.model == "Qwen/Qwen3-4B" and args.layer != -5:
        raise ValueError("Frozen proportional layer for Qwen3-4B is -5")

    df = pd.read_csv(CSV_PATH).copy()

    word_col = find_col(df, ["word", "token"])
    sent_col = find_col(df, ["snt_id", "sent_id", "sentence_id"])
    onset_col = find_col(df, ["onset", "onset_sec", "word_onset", "start"])
    offset_col = find_col(df, ["offset", "offset_sec", "word_offset", "end"])
    run_col = find_col(df, ["run_id", "run"])

    df[word_col] = df[word_col].astype(str).str.strip()

    df = df.sort_values(
        [run_col, sent_col, onset_col, offset_col]
    ).reset_index(drop=True)

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

    if device != "cuda":
        raise RuntimeError("Qwen3-4B extraction requires a CUDA GPU for this workflow")
    model.to(device)
    model.eval()

    print("Device:", device)
    print("Layer:", args.layer)
    assert model.config.num_hidden_layers == 36, "Unexpected Qwen3-4B architecture"

    rows = []
    token_id = 1
    n_missing = 0

    for run_id, g in df.groupby(run_col, sort=True):

        g = g.sort_values(
            [sent_col, onset_col, offset_col]
        ).reset_index(drop=True)

        words = g[word_col].tolist()

        print(
            f"\nRun {run_id}: "
            f"{len(words)} words"
        )

        enc = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=True,
        )

        word_ids = enc.word_ids(batch_index=0)

        n_subtokens = enc["input_ids"].shape[1]

        print(
            f"Run {run_id}: "
            f"{n_subtokens} subtokens"
        )

        enc_device = {
            k: v.to(device)
            for k, v in enc.items()
        }

        with torch.no_grad():
            out = model(
                **enc_device,
                output_hidden_states=True,
                use_cache=False,
            )

        H = (
            out.hidden_states[args.layer][0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        if H.shape[0] != n_subtokens:
            raise RuntimeError(
                f"Hidden-state/token length mismatch "
                f"in run {run_id}: "
                f"{H.shape[0]} vs {n_subtokens}"
            )

        for local_i, row in g.iterrows():

            sub_idx = [
                j
                for j, wid in enumerate(word_ids)
                if wid == local_i
            ]

            if len(sub_idx) == 0:
                n_missing += 1
                print(
                    f"[missing] run={run_id} "
                    f"word={local_i} "
                    f"text={row[word_col]!r}"
                )
                continue

            # Match original Qwen analysis exactly:
            # mean pooling over all subtokens belonging to the word.
            emb = (
                H[sub_idx, :]
                .mean(axis=0)
                .astype(np.float32)
            )

            rows.append({
                "word": str(row[word_col]),
                "snt_id": int(row[sent_col]),
                "token_id": token_id,
                "onset": float(row[onset_col]),
                "offset": float(row[offset_col]),
                "run_id": int(row[run_col]),
                "n_subtokens": len(sub_idx),
                "embedding": emb,
            })

            token_id += 1

        # Release run-sized tensors before next run
        del out
        del H
        del enc_device

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"Completed run {run_id}. "
            f"Total embedded words: {len(rows)}"
        )

    out_df = pd.DataFrame(rows)

    out_path = out_dir / args.out_name
    out_df.to_pickle(out_path)

    X = np.vstack(out_df["embedding"].values)

    print("\nDone.")
    print("Output:", out_path)
    print("DataFrame shape:", out_df.shape)
    print("Embedding matrix:", X.shape)
    print("Missing words:", n_missing)
    print("NaNs:", int(np.isnan(X).sum()))
    print("Runs:", sorted(out_df["run_id"].unique()))


if __name__ == "__main__":
    main()