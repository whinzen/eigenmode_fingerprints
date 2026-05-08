# ~/eigenmode_fingerprints/code/build_sent_bound_per_run.py
import argparse, json
from pathlib import Path
import pandas as pd
from settings import CSV_ANN, PANG_OUT, RUN_OFFSET, BOUND_JSON

def main(csv, out_json, unit):
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv)
    cols = {c.lower().strip(): c for c in df.columns}
    need = ["snt_id","token_id","onset","offset","run_id"]
    missing = [n for n in need if n not in cols]
    if missing:
        raise SystemExit(f"CSV is missing {missing}. Found: {list(df.columns)}")

    df = df.rename(columns={
        cols["snt_id"]: "snt_id",
        cols["token_id"]: "token_id",
        cols["onset"]:   "onset",
        cols["offset"]:  "offset",
        cols["run_id"]:  "run_id"
    })
    # unit conversion
    if unit == "auto":
        # Heuristic: if typical values > 100 (ms-like), treat as ms
        guess_ms = (df["onset"].median() > 100) or (df["offset"].median() > 100)
        factor = 0.001 if guess_ms else 1.0
        in_unit = "ms" if guess_ms else "sec"
    elif unit == "ms":
        factor, in_unit = 0.001, "ms"
    else:
        factor, in_unit = 1.0, "sec"

    df["onset_sec"]  = pd.to_numeric(df["onset"], errors="coerce") * factor
    df["offset_sec"] = pd.to_numeric(df["offset"], errors="coerce") * factor
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce").astype("Int64")

    # End-of-sentence = max(offset) per (run_id, snt_id)
    ends = (
        df.dropna(subset=["run_id","snt_id","offset_sec"])
          .groupby(["run_id","snt_id"])["offset_sec"].max().reset_index()
    )

    boundaries = {}
    for r in sorted(ends["run_id"].dropna().unique()):
        rr = int(r) + RUN_OFFSET  # map 1..9 → 15..23 (typically)
        lbl = f"run-{rr}"
        times = sorted(set(round(float(t), 3) for t in ends.loc[ends["run_id"]==r, "offset_sec"]))
        boundaries[lbl] = times

    with open(out_json, "w") as f:
        json.dump(boundaries, f, indent=2)

    print(f"Wrote boundary times per run to: {out_json}")
    for run in sorted(boundaries):
        print(f"  {run}: {len(boundaries[run])} boundaries  (input_unit={in_unit}, run_offset=+{RUN_OFFSET})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV_ANN))
    ap.add_argument("--out-json", default=str(BOUND_JSON))
    ap.add_argument("--unit", choices=["auto","ms","sec"], default="auto")
    args = ap.parse_args()
    main(args.csv, args.out_json, args.unit)