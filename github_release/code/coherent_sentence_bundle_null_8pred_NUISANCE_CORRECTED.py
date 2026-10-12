#!/usr/bin/env python
# NUISANCE-CORRECTED coherent sentence onset+pause null.
# Preserves historical full-grid circular shift; trims initial nonsteady TRs only after shifting.
from pathlib import Path
import time, numpy as np, pandas as pd

N_PERM=10000; SEED=20260915; MIN_TRS=100; MIN_SHIFT_TR=20
EXPECTED_RUNS=439; EXPECTED_SUBJECTS=49; EXPECTED_HEMI_RUNS=878
HIST_ONSET_ABS=0.07166698610960608; HIST_ONSET_RMS=0.07295713899857484
BASE=Path.home()/"eigenmode_fingerprints"; PANG=BASE/"pang_out"
MODAL_ROOT=PANG/"nuisance_robustness"/"corrected_modal_amplitudes"
REG_ROOT=PANG/"regressors_sentence_controls_exact"
AUTH_FILE=PANG/"nuisance_robustness"/"corrected_validated_linguistic_effects"/"corrected_sentence_effects_run_level_FIXED.csv"
OUT=PANG/"nuisance_robustness"/"figure3_corrected"/"coherent_sentence_bundle_null_8pred"; OUT.mkdir(parents=True,exist_ok=True)
ONSET=REG_ROOT/"sentence_onset_hrf"; PAUSE=REG_ROOT/"preceding_pause_centered_hrf"
FIXED={"word_rate":REG_ROOT/"word_rate_hrf","content_density":REG_ROOT/"content_density_hrf",
"sentence_shift":REG_ROOT/"sentence_shift_hrf","acoustic_envelope":REG_ROOT/"acoustic_envelope_hrf",
"mean_log_f0":REG_ROOT/"mean_log_f0_hrf","pitch_reset":REG_ROOT/"sentence_pitch_reset_hrf"}
ORDER=list(FIXED)

def parse(f):
 s=Path(f).stem; a,b=s.rsplit("_run-",1); return a,int(b)
def loadA(s,r,h):
 for f in [MODAL_ROOT/s/f"run-{r}"/f"A_{h}.npy",MODAL_ROOT/s/f"run-{r:02d}"/f"A_{h}.npy"]:
  if f.exists():
   A=np.asarray(np.load(f),float)
   if A.ndim!=2: raise RuntimeError(f"bad A {f}: {A.shape}")
   if A.shape[0]==120:return A
   if A.shape[1]==120:return A.T
   raise RuntimeError(f"no 120-mode axis {f}: {A.shape}")
 return None
def resid(Q,X): return X-Q@(Q.T@X)
def fit(d,on,pa):
 G=np.column_stack([on,pa]); Gr=resid(d["Q"],G)
 try: inv=np.linalg.inv(Gr.T@Gr)
 except np.linalg.LinAlgError:return None
 so=np.std(on,ddof=1); sp=np.std(pa,ddof=1); z={}
 for h in ["L","R"]:
  B=inv@(Gr.T@d["hemi"][h]["Yr"]); sy=d["hemi"][h]["sy"]
  z[h]={"bo":B[0],"bp":B[1],"bso":B[0]*so/sy,"bsp":B[1]*sp/sy}
 return z
def aggregate(ds,fits,subs):
 ix={s:i for i,s in enumerate(subs)}; os=np.zeros((len(subs),2,119)); ps=np.zeros_like(os); n=np.zeros((len(subs),2),int)
 for d in ds:
  z=fits[(d["subject"],d["run"])]
  for hi,h in enumerate(["L","R"]):
   os[ix[d["subject"]],hi]+=z[h]["bso"][1:]; ps[ix[d["subject"]],hi]+=z[h]["bsp"][1:]; n[ix[d["subject"]],hi]+=1
 if np.any(n==0): raise RuntimeError("zero participant/hemi count")
 return np.mean(np.mean(os/n[:,:,None],axis=0),axis=0),np.mean(np.mean(ps/n[:,:,None],axis=0),axis=0)
def stats(o,p):
 return {"onset_mean_beta_star":float(np.mean(o)),"onset_mean_abs_beta_star":float(np.mean(np.abs(o))),
 "onset_profile_rms":float(np.sqrt(np.mean(o**2))),"pause_mean_beta_star":float(np.mean(p)),
 "pause_mean_abs_beta_star":float(np.mean(np.abs(p))),"bundle_rms_beta_star":float(np.sqrt(np.mean(np.r_[o,p]**2)))}

auth=pd.read_csv(AUTH_FILE); auth["run"]=auth.run.astype(int); auth["mode"]=auth["mode"].astype(int)
ao=auth[auth.predictor.eq("Sentence onset")].copy()
grid=ao[["subject","run"]].drop_duplicates(); keys=set(map(tuple,grid.to_numpy()))
assert len(keys)==439 and grid.subject.nunique()==49
assert ao[["subject","run","hemisphere"]].drop_duplicates().shape[0]==878
assert len(ao)==439*2*120
print("Authoritative corrected grid:",len(keys),"runs,",grid.subject.nunique(),"participants, 878 hemisphere-runs")

rec=[]
for f in sorted(ONSET.glob("sub-*_run-*.npy")):
 s,r=parse(f.name); rec.append({"file":f.name,"subject":s,"run":r})
m=pd.DataFrame(rec).sort_values(["subject","run"]); m["stimulus_run"]=m.groupby("subject").cumcount()+1
m=m[m.stimulus_run.between(1,9)].copy(); m["key"]=list(zip(m.subject,m.run))
m=m[m.key.isin(keys)].copy()
if set(m.key)!=keys: raise RuntimeError(f"mapping mismatch missing={sorted(keys-set(m.key))}")

ds=[]; align=[]
for _,q in m.sort_values(["subject","run"]).iterrows():
 fn=q.file; s=q.subject; r=int(q.run); sr=int(q.stimulus_run)
 of=np.asarray(np.load(ONSET/fn),float).squeeze(); pf=np.asarray(np.load(PAUSE/fn),float).squeeze(); Tf=len(of)
 if len(pf)!=Tf: raise RuntimeError(f"pause mismatch {fn}")
 ff={}
 for k in ORDER:
  x=np.asarray(np.load(FIXED[k]/fn),float).squeeze()
  if len(x)!=Tf: raise RuntimeError(f"{k} mismatch {fn}")
  ff[k]=x
 AL,AR=loadA(s,r,"L"),loadA(s,r,"R")
 if AL is None or AR is None or AL.shape!=AR.shape: raise RuntimeError(f"A missing/mismatch {s} {r}")
 Tc=AL.shape[1]; nr=Tf-Tc
 if nr<0 or nr>10 or Tc<MIN_TRS: raise RuntimeError(f"bad trim {s} {r}: {Tf}->{Tc}")
 on,pa=of[nr:],pf[nr:]; X0=np.column_stack([np.ones(Tc)]+[ff[k][nr:] for k in ORDER]); Q,_=np.linalg.qr(X0,mode="reduced")
 hd={}
 for h,A in [("L",AL),("R",AR)]:
  Y=(A**2).T
  if Y.shape!=(Tc,120): raise RuntimeError(f"Y shape {s} {r} {h} {Y.shape}")
  hd[h]={"Yr":resid(Q,Y),"sy":np.std(Y,axis=0,ddof=1)}
 ds.append({"subject":s,"run":r,"stimulus_run":sr,"T_full":Tf,"T":Tc,"n_removed":nr,"onset_full":of,"pause_full":pf,"onset":on,"pause":pa,"Q":Q,"hemi":hd})
 align.append({"subject":s,"run":r,"stimulus_run":sr,"T_full":Tf,"T_corrected":Tc,"n_removed":nr})
if set((d["subject"],d["run"]) for d in ds)!=keys: raise RuntimeError("loaded grid mismatch")
subs=sorted(set(d["subject"] for d in ds))
assert len(ds)==439 and len(subs)==49
ad=pd.DataFrame(align); ad.to_csv(OUT/"corrected_sentence_null_alignment_QC.csv",index=False)
print("Loaded corrected grid: 439 runs, 49 participants, 878 hemisphere-runs")
print("Removed initial TRs:\n",ad.n_removed.value_counts().sort_index().to_string())

ef={}; rr=[]
for d in ds:
 z=fit(d,d["onset"],d["pause"])
 if z is None: raise RuntimeError(f"singular empirical fit {d['subject']} {d['run']}")
 ef[(d["subject"],d["run"])]=z
 for h in ["L","R"]:
  for mode,b in enumerate(z[h]["bo"]): rr.append({"subject":d["subject"],"run":d["run"],"hemisphere":h,"mode":mode,"beta_reconstructed":b})
rr=pd.DataFrame(rr)
cmp=ao[["subject","run","hemisphere","mode","beta"]].rename(columns={"beta":"beta_authoritative"}).merge(rr,on=["subject","run","hemisphere","mode"],how="outer",validate="one_to_one",indicator=True)
if not cmp._merge.eq("both").all(): raise RuntimeError("beta grids mismatch")
a=cmp.beta_authoritative.to_numpy(float); b=cmp.beta_reconstructed.to_numpy(float)
br=float(np.corrcoef(a,b)[0,1]); md=float(np.max(np.abs(a-b))); mad=float(np.mean(np.abs(a-b)))
print(f"Raw onset-beta reconstruction: r={br:.12f}, maxdiff={md:.12e}, meandiff={mad:.12e}")
cmp.drop(columns="_merge").to_csv(OUT/"corrected_onset_raw_beta_reconstruction_QC.csv",index=False)
scale=max(1.,float(np.max(np.abs(a))))
if br<.999999999 or md>1e-8*scale: raise RuntimeError("FATAL QC: reconstructed raw onset betas do not match authoritative corrected betas")

eo,ep=aggregate(ds,ef,subs); es=stats(eo,ep)
print("\nCorrected empirical statistics:")
for k,v in es.items(): print(f" {k:30s}: {v:.12f}")
print(f"\nHistorical onset mean |beta*| : {HIST_ONSET_ABS:.12f}")
print(f"Corrected  onset mean |beta*| : {es['onset_mean_abs_beta_star']:.12f}")
print(f"Ratio corrected/historical   : {es['onset_mean_abs_beta_star']/HIST_ONSET_ABS:.6f}")
print(f"Historical onset RMS          : {HIST_ONSET_RMS:.12f}")
print(f"Corrected  onset RMS          : {es['onset_profile_rms']:.12f}")
print(f"Ratio corrected/historical   : {es['onset_profile_rms']/HIST_ONSET_RMS:.6f}")
pd.DataFrame({"mode":np.arange(1,120),"onset_beta_star":eo,"pause_beta_star":ep}).to_csv(OUT/"corrected_empirical_onset_pause_beta_star_profiles.csv",index=False)
pd.DataFrame([{**es,"historical_onset_mean_abs_beta_star":HIST_ONSET_ABS,"historical_onset_profile_rms":HIST_ONSET_RMS,"raw_beta_reconstruction_r":br,"raw_beta_reconstruction_max_abs_diff":md,"n_runs":439,"n_subjects":49}]).to_csv(OUT/"corrected_empirical_onset_pause_summary.csv",index=False)

print(f"\nRunning {N_PERM} QC permutations...")
ix={s:i for i,s in enumerate(subs)}; results=[]; ops=[]; pps=[]; t0=time.time()
for perm in range(N_PERM):
 t=time.time(); rng=np.random.default_rng(SEED+perm); lags={}
 for sr in range(1,10):
  Ts=[d["T_full"] for d in ds if d["stimulus_run"]==sr]
  if Ts:
   poss=np.arange(MIN_SHIFT_TR,min(Ts)-MIN_SHIFT_TR)
   if not len(poss): raise RuntimeError(f"no valid lag section {sr}")
   lags[sr]=int(rng.choice(poss))
 os=np.zeros((49,2,119)); ps=np.zeros_like(os); n=np.zeros((49,2),int)
 for d in ds:
  lag=lags[d["stimulus_run"]]; nr=d["n_removed"]
  on=np.roll(d["onset_full"],lag)[nr:]; pa=np.roll(d["pause_full"],lag)[nr:]
  z=fit(d,on,pa)
  if z is None: continue
  si=ix[d["subject"]]
  for hi,h in enumerate(["L","R"]):
   os[si,hi]+=z[h]["bso"][1:]; ps[si,hi]+=z[h]["bsp"][1:]; n[si,hi]+=1
 if np.any(n==0): raise RuntimeError(f"zero count perm {perm}")
 op=np.mean(np.mean(os/n[:,:,None],axis=0),axis=0); pp=np.mean(np.mean(ps/n[:,:,None],axis=0),axis=0)
 st=stats(op,pp); st.update({"perm":perm,"onset_r_empirical":float(np.corrcoef(op,eo)[0,1]),"pause_r_empirical":float(np.corrcoef(pp,ep)[0,1]),"seconds":time.time()-t})
 results.append(st); ops.append(op); pps.append(pp)
 print(f"{perm+1:2d}/{N_PERM}: onset |beta*|={st['onset_mean_abs_beta_star']:.6f}, RMS={st['onset_profile_rms']:.6f}, {st['seconds']:.2f}s")
rdf=pd.DataFrame(results)
final=OUT/f"coherent_sentence_bundle_null_corrected_{N_PERM}.csv"; rdf.to_csv(final,index=False)
np.save(OUT/f"coherent_sentence_bundle_null_corrected_{N_PERM}_onset_profiles.npy",np.asarray(ops))
np.save(OUT/f"coherent_sentence_bundle_null_corrected_{N_PERM}_pause_profiles.npy",np.asarray(pps))
rows=[]
for metric,emp in [("onset_mean_abs_beta_star",es["onset_mean_abs_beta_star"]),("onset_profile_rms",es["onset_profile_rms"]),("pause_mean_abs_beta_star",es["pause_mean_abs_beta_star"]),("bundle_rms_beta_star",es["bundle_rms_beta_star"])]:
 null=rdf[metric].to_numpy(float); rows.append({"metric":metric,"empirical":emp,"null_mean":np.mean(null),"null_sd":np.std(null,ddof=1),"null_2p5":np.quantile(null,.025),"null_97p5":np.quantile(null,.975),"p_perm_greater_QC_ONLY":(1+np.sum(null>=emp))/(len(null)+1)})
summ=pd.DataFrame(rows); sf=OUT/f"coherent_sentence_bundle_null_corrected_{N_PERM}_summary.csv"; summ.to_csv(sf,index=False)
print("\nQC RUN COMPLETE — N_PERM=10, p-values are NOT inferential.")
print(summ.round(8).to_string(index=False))
print("Elapsed minutes:",(time.time()-t0)/60)
print("Saved:",final,sf,sep="\n")
