# ~/eigenmode_fingerprints/code/report_trs.py
from pathlib import Path
import json
import pandas as pd
import nibabel as nib
from settings import EMP_DIR, PANG_OUT

OUT_JSON = PANG_OUT/"group"/"tr_by_subject_and_run.json"
OUT_TSV  = PANG_OUT/"group"/"tr_by_subject_and_run.tsv"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

rows = []
for sdir in sorted(EMP_DIR.glob("sub-*")):
    sub = sdir.name
    fgif = sorted(sdir.glob("func/*_hemi-L_space-fsaverage5_bold.func.gii"))
    if not fgif:
        continue
    seen = []
    for f in fgif:
        run = int(f.name.split("_run-")[1].split("_")[0])
        img = nib.load(str(f))
        tr = float(img.darrays[0].metadata.get("TimeStep", "nan"))
        rows.append({"subject": sub, "run": run, "TR_s": tr})
        seen.append(run)
    print(f"✅ {sub}: unique TRs (s) = {sorted(set([r['TR_s'] for r in rows if r['subject']==sub]))}")

df = pd.DataFrame(rows).sort_values(["subject","run"])
df.to_json(OUT_JSON, orient="records", indent=2)
df.to_csv(OUT_TSV, sep="\t", index=False)
print(f"Wrote: {OUT_JSON}\nWrote: {OUT_TSV}")