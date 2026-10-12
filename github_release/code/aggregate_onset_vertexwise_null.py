#!/usr/bin/env python
"""Aggregate onset vertexwise structured-null chunks and create inferential maps."""
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib

BASE=Path.home()/"eigenmode_fingerprints"
ROOT=BASE/"pang_out"/"nuisance_robustness"/"vertex_space_linguistic_encoding"
SRC=ROOT/"structured_null_qc"
OUT=ROOT/"onset_vertexwise_null_10000"

csvs=sorted(OUT.glob("maxabs_*.csv"))
if len(csvs)!=100:
    raise RuntimeError(f"Expected 100 maxabs chunks, found {len(csvs)}")

mx=pd.concat([pd.read_csv(f) for f in csvs],ignore_index=True).sort_values("perm")
assert len(mx)==10000
assert mx["perm"].nunique()==10000
assert set(mx["perm"].astype(int))==set(range(10000))
maxnull=mx["max_abs_beta_star"].to_numpy(float)

summary=[]
maps={}
for h in ["L","R"]:
    emp=np.load(SRC/f"sentence_onset_empirical_group_beta_star_{h}.npy").astype(float)
    fs=sorted(OUT.glob(f"exceed_{h}_*.npy"))
    if len(fs)!=100: raise RuntimeError(f"Expected 100 {h} count chunks, found {len(fs)}")
    counts=np.sum([np.load(f).astype(np.int64) for f in fs],axis=0)
    assert counts.shape==emp.shape==(10242,)
    p_unc=(1+counts)/(10000+1)
    ae=np.abs(emp)
    # max-stat FWER: compare each observed vertex to whole-cortex null maximum.
    p_fwer=(1+(maxnull[:,None]>=ae[None,:]).sum(axis=0))/(10000+1)
    sig=p_fwer<=0.05
    maps[h]=(emp,p_unc,p_fwer,sig)
    summary.append({
        "hemisphere":h,
        "n_vertices":len(emp),
        "n_FWER_p_le_0p05":int(sig.sum()),
        "min_uncorrected_p":float(p_unc.min()),
        "min_FWER_p":float(p_fwer.min()),
        "empirical_RMS":float(np.sqrt(np.mean(emp**2))),
        "empirical_mean":float(np.mean(emp)),
        "empirical_mean_abs":float(np.mean(np.abs(emp))),
    })

    def save_metric(name,arr):
        img=nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(np.asarray(arr,np.float32))])
        nib.save(img,OUT/f"sentence_onset_{name}_{h}.func.gii")
    save_metric("group_beta_star",emp)
    save_metric("p_uncorrected",p_unc)
    save_metric("p_FWER_maxstat",p_fwer)
    save_metric("FWER_sig_mask",sig.astype(float))

pd.DataFrame(summary).to_csv(OUT/"sentence_onset_vertexwise_summary.csv",index=False)
mx.to_csv(OUT/"sentence_onset_maxabs_null_10000.csv",index=False)

# Descriptive lateralization (NO inferential p-value).
rmsL=summary[0]["empirical_RMS"]; rmsR=summary[1]["empirical_RMS"]
lat=pd.DataFrame([{
    "RMS_L":rmsL,"RMS_R":rmsR,"RMS_L_minus_R":rmsL-rmsR,
    "RMS_L_over_R":rmsL/rmsR
}])
lat.to_csv(OUT/"sentence_onset_descriptive_lateralization.csv",index=False)

print(pd.DataFrame(summary).to_string(index=False))
print("\nDescriptive lateralization:")
print(lat.to_string(index=False))
print("\nNull max |beta*| 95th percentile:",np.quantile(maxnull,.95))
print("Null max |beta*| 97.5th percentile:",np.quantile(maxnull,.975))
print("\nWrote:",OUT)
print("FWER inference is two-sided whole-cortex max-|beta*| corrected.")
