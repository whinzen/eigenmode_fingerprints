#!/usr/bin/env python
from pathlib import Path
import argparse, gc, time
import numpy as np, pandas as pd
from scipy.stats import gamma as gamma_dist
from scipy.special import gamma as gamma_func
P=argparse.ArgumentParser(); P.add_argument('--start',type=int,required=True); P.add_argument('--n',type=int,default=100); P.add_argument('--qc',action='store_true'); a=P.parse_args()
BASE=Path.home()/"eigenmode_fingerprints"; PANG=BASE/"pang_out"
ROOT=PANG/"nuisance_robustness"/"vertex_space_linguistic_encoding"; CACHE=ROOT/"corrected_vertex_cache"; OUT=ROOT/"structured_null_10000"; OUT.mkdir(parents=True,exist_ok=True)
REG=PANG/"regressors_sentence_controls_exact"; TOKREG=PANG/"regressors_qwen3_0p6b_continuous_centered"
SENT={"sentence_onset":REG/"sentence_onset_hrf","word_rate":REG/"word_rate_hrf","content_density":REG/"content_density_hrf","preceding_pause":REG/"preceding_pause_centered_hrf","sentence_shift":REG/"sentence_shift_hrf","acoustic_envelope":REG/"acoustic_envelope_hrf","mean_log_f0":REG/"mean_log_f0_hrf","pitch_reset":REG/"sentence_pitch_reset_hrf"}
ORDER=list(SENT); ion=ORDER.index('sentence_onset'); ipa=ORDER.index('preceding_pause'); ish=ORDER.index('sentence_shift')
manifest=pd.read_csv(OUT/"vertex_cache_manifest.csv"); manifest.run=manifest.run.astype(int); manifest.stimulus_run=manifest.stimulus_run.astype(int)
subs=sorted(manifest.subject.unique()); nr=manifest.groupby('subject').size().to_dict(); Nsub=len(subs); assert len(manifest)==439 and Nsub==49
end=min(a.start+a.n,10000); perms=np.arange(a.start,end); B=len(perms); assert B>0
spec=np.load(OUT/"surrogate_specs_10000.npz"); onset_lags=spec['onset_lags'][perms]; token_lags=spec['token_lags'][perms]
# events
shift_events=pd.read_csv(PANG/"sentence_level_controls"/"sentence_shift_qwen_continuous.csv"); rc='stimulus_run' if 'stimulus_run' in shift_events else 'run_id'; oc='sentence_onset' if 'sentence_onset' in shift_events else 'onset'; shift_events['shift_centered']=shift_events.sentence_shift-shift_events.groupby(rc).sentence_shift.transform('mean')
sev={int(sr):(d.sort_values(oc)[oc].to_numpy(float),d.sort_values(oc).shift_centered.to_numpy(float)) for sr,d in shift_events.groupby(rc)}
trans=pd.read_csv(PANG/"word_transition_geometry_qwen3_0p6b_continuous"/"word_transition_geometry.csv"); surp=pd.read_csv(PANG/"token_surprisal_qwen_continuous"/"token_surprisal_qwen3_0p6b_continuous.csv")
if 'same_run' in trans: trans=trans.loc[trans.same_run.eq(1)].copy()
trans.run_id=trans.run_id.astype(int); surp.run_id=surp.run_id.astype(int)
tev={}
for name,tab,col in [('token_surprisal',surp,'surprisal_sum'),('token_curvature',trans,'curvature')]:
    tev[name]={}
    for sr in range(1,10):
        d=tab.loc[tab.run_id.eq(sr)].sort_values('onset'); on=d.onset.to_numpy(float); v=d[col].to_numpy(float); g=np.isfinite(on)&np.isfinite(v); tev[name][sr]=(on[g],v[g])
TR=2.; t=np.arange(0,32,TR); Hs=gamma_dist.pdf(t,6)-.5*gamma_dist.pdf(t,16); Hs/=np.max(np.abs(Hs))
def gp(x,k=6):
    o=np.zeros_like(x,float); m=x>0; o[m]=x[m]**(k-1)*np.exp(-x[m])/gamma_func(k); return o
Ht=gp(t,6)-(1/6)*gp(t,16); Ht/=np.max(np.abs(Ht))
def z(x,ddof=0):
    sd=np.std(x,ddof=ddof); return np.zeros_like(x) if (not np.isfinite(sd) or sd==0) else (x-np.mean(x))/sd
def loadreg(folder,s,r):
    for f in (folder/f"{s}_run-{r}.npy",folder/f"{s}_run-{r:02d}.npy"):
        if f.exists(): return np.asarray(np.load(f),float).squeeze()
    raise FileNotFoundError((folder,s,r))
def build_shift(T,sr,w):
    x=np.zeros(T); on,_=sev[sr]
    for q,v in zip(on,w):
        j=int(np.round(q/TR));
        if 0<=j<T: x[j]+=v
    return z(np.convolve(x,Hs,'full')[:T],ddof=1)
def build_token(T,pred,sr,lag):
    on,v=tev[pred][sr]; v=np.roll(v,int(lag)); v=v-v.mean(); x=np.zeros(T)
    for q,u in zip(on,v):
        j=int(round(float(q)/TR));
        if 0<=j<T: x[j]+=u
    return z(np.convolve(x,Ht,'full')[:T],ddof=0)
def qr_resid(Q,G): return G-Q@(Q.T@G)
def single_batch(Y,G,fixed):
    # G T x B; beta* B x V
    Q,_=np.linalg.qr(np.column_stack([np.ones(len(Y)),fixed]),mode='reduced'); R=qr_resid(Q,G); den=np.sum(R*R,axis=0); Bm=(R.T@Y)/den[:,None]; sx=np.std(G,axis=0,ddof=1); return Bm*sx[:,None]
def onset_batch(Y,O,A,fixed):
    Q,_=np.linalg.qr(np.column_stack([np.ones(len(Y)),fixed]),mode='reduced'); R1=qr_resid(Q,O); R2=qr_resid(Q,A); aa=np.sum(R1*R1,0); bb=np.sum(R1*R2,0); cc=np.sum(R2*R2,0); det=aa*cc-bb*bb; D=R1.T@Y; E=R2.T@Y; bo=(cc[:,None]*D-bb[:,None]*E)/det[:,None]; return bo*np.std(O,axis=0,ddof=1)[:,None]
# B x V group accumulators, lazy by pred/hemi
preds=['sentence_onset','sentence_shift','token_surprisal','token_curvature']; acc={p:{'L':None,'R':None} for p in preds}
t0=time.time()
for ii,row in manifest.iterrows():
    s=str(row.subject); r=int(row.run); sr=int(row.stimulus_run); T=int(row.T_full); keep=np.load(CACHE/s/f"run-{r}"/"keep.npy").astype(bool)
    sf={k:loadreg(v,s,r) for k,v in SENT.items()}; X=np.column_stack([sf[k][keep] for k in ORDER]); fixed_on=X[:,[j for j in range(8) if j not in (ion,ipa)]]; fixed_sh=X[:,[j for j in range(8) if j!=ish]]
    # onset full-grid shifts
    O=np.column_stack([np.roll(sf['sentence_onset'],int(onset_lags[b,sr-1]))[keep] for b in range(B)]); A=np.column_stack([np.roll(sf['preceding_pause'],int(onset_lags[b,sr-1]))[keep] for b in range(B)])
    # shift exact RNG(SEED+perm), permutations of weights within sections; only current sr retained
    _,native_w=sev[sr]; GS=[]
    for pidx in perms:
        rng=np.random.default_rng(20260914+int(pidx)); w=None
        for ss in range(1,10):
            ww=rng.permutation(sev[ss][1])
            if ss==sr: w=ww
        GS.append(build_shift(T,sr,w)[keep])
    GS=np.column_stack(GS)
    # token historical lag indices 0=surprisal,1=curvature
    GU=np.column_stack([build_token(T,'token_surprisal',sr,token_lags[b,0,sr-1])[keep] for b in range(B)])
    GC=np.column_stack([build_token(T,'token_curvature',sr,token_lags[b,1,sr-1])[keep] for b in range(B)])
    wrs=loadreg(TOKREG/'word_rate_hrf_per_subject',s,r)[keep,None]
    weight=1.0/nr[s]/Nsub
    for h in ['L','R']:
        Y=np.load(CACHE/s/f"run-{r}"/f"Z_{h}.npy",mmap_mode='r').T
        V=Y.shape[1]
        for p in preds:
            if acc[p][h] is None: acc[p][h]=np.zeros((B,V),np.float64)
        acc['sentence_onset'][h]+=weight*onset_batch(Y,O,A,fixed_on)
        acc['sentence_shift'][h]+=weight*single_batch(Y,GS,fixed_sh)
        acc['token_surprisal'][h]+=weight*single_batch(Y,GU,wrs)
        acc['token_curvature'][h]+=weight*single_batch(Y,GC,wrs)
        del Y; gc.collect()
    if (ii+1)%50==0: print(f"{ii+1}/439 runs | {(time.time()-t0)/60:.1f} min",flush=True)
# ============================================================
# ONSET VERTEXWISE INFERENCE OUTPUT
#
# IMPORTANT: everything above this block is unchanged from the
# previously validated 4-predictor worker.
# ============================================================

qcroot=ROOT/'structured_null_qc'
vout=ROOT/'onset_vertexwise_null_10000'
vout.mkdir(parents=True,exist_ok=True)

# Existing observed group onset maps from notebook 07.
emp={
    h:np.load(qcroot/f"sentence_onset_empirical_group_beta_star_{h}.npy").astype(float)
    for h in ['L','R']
}

# Retain the historical scalar RMS output for onset as an additional
# provenance check against the already completed 10,000-permutation run.
rms=np.sqrt(
    (np.sum(acc['sentence_onset']['L']**2,axis=1)
     +np.sum(acc['sentence_onset']['R']**2,axis=1))
    /(acc['sentence_onset']['L'].shape[1]+acc['sentence_onset']['R'].shape[1])
)
pd.DataFrame({
    'perm':perms.astype(int),
    'predictor':'sentence_onset',
    'group_map_rms':rms
}).to_csv(vout/f"rms_{a.start:05d}_{end-1:05d}.csv",index=False)

# Whole-cortex two-sided max-|beta*| statistic.
maxabs=np.maximum(
    np.max(np.abs(acc['sentence_onset']['L']),axis=1),
    np.max(np.abs(acc['sentence_onset']['R']),axis=1)
)
pd.DataFrame({
    'perm':perms.astype(int),
    'max_abs_beta_star':maxabs
}).to_csv(vout/f"maxabs_{a.start:05d}_{end-1:05d}.csv",index=False)

# Per-vertex two-sided uncorrected exceedance counts.
for h in ['L','R']:
    counts=np.sum(
        np.abs(acc['sentence_onset'][h]) >= np.abs(emp[h])[None,:],
        axis=0
    ).astype(np.int32)
    np.save(vout/f"exceed_{h}_{a.start:05d}_{end-1:05d}.npy",counts)

# QC:
# 1) reproduce the exact scalar RMS criterion used by the validated worker;
# 2) report map-level numerical discrepancy as a diagnostic only.
#
# The original optimized worker was validated at the RMS level, not by
# bitwise/mapwise equality. Batched BLAS changes floating-point operation
# order, producing tiny (~1e-10) vertexwise differences while preserving
# the inferential statistic.
if a.qc:
    if a.start!=0 or end<10:
        raise RuntimeError("--qc requires a range containing permutations 0..9")

    Lref=np.load(qcroot/"sentence_onset_null_group_beta_star_L_10perm.npy")
    Rref=np.load(qcroot/"sentence_onset_null_group_beta_star_R_10perm.npy")
    ref_rms=np.sqrt(
        (np.sum(Lref**2,axis=1)+np.sum(Rref**2,axis=1))
        /(Lref.shape[1]+Rref.shape[1])
    )
    got_rms=rms[:10]
    rmsdiff=float(np.max(np.abs(ref_rms-got_rms)))
    rmsok=bool(np.allclose(ref_rms,got_rms,rtol=1e-10,atol=1e-12))

    print("\nQC against notebook 10-perm onset null:")
    print("RMS max_abs_diff=",rmsdiff,"allclose=",rmsok)

    for h,ref in [('L',Lref),('R',Rref)]:
        d=float(np.max(np.abs(ref-acc['sentence_onset'][h][:10])))
        corr=np.corrcoef(ref.ravel(),acc['sentence_onset'][h][:10].ravel())[0,1]
        print(h,"map max_abs_diff=",d,"map corr=",corr)

    if not rmsok:
        raise RuntimeError("Validated RMS QC failed")

print('Wrote onset vertexwise chunk',a.start,end-1,'to',vout)
