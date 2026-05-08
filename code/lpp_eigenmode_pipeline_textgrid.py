# lpp_eigenmode_pipeline_textgrid.py
# -------------------------------------------------------
# Naturalistic language fMRI (Le Petit Prince, EN)
# - Reads Praat TextGrids (annotation/EN/lppEN_section*.TextGrid)
# - Projects preprocessed BOLD to cortical Laplace–Beltrami eigenmodes
# - Computes mode-energy spectrum and ERME around sentence boundaries
# Requires: numpy, pandas, nibabel, nilearn, scipy
# -------------------------------------------------------

import re
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import nibabel as nib
from nilearn import surface, datasets
from nilearn.image import index_img
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import zscore


# -------------------------------
# Data structures
# -------------------------------

@dataclass
class SurfMesh:
    coords: np.ndarray
    faces: np.ndarray
    areas: np.ndarray
    hemi: str  # "L" or "R"


@dataclass
class Eigenmodes:
    Phi: np.ndarray        # (V, K) eigenvectors (DC excluded)
    lam: np.ndarray        # (K,) eigenvalues (DC excluded)
    weights: np.ndarray    # (V,) vertex areas
    hemi: str


# -------------------------------
# Mesh & Laplacian utilities
# -------------------------------

def _triangle_areas(coords: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = coords[faces[:, 0]]
    v1 = coords[faces[:, 1]]
    v2 = coords[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _vertex_areas(coords: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri_areas = _triangle_areas(coords, faces)
    areas = np.zeros(coords.shape[0])
    for i in range(3):
        np.add.at(areas, faces[:, i], tri_areas / 3.0)
    return areas


def _cotangent_laplacian(coords: np.ndarray, faces: np.ndarray):
    V = coords.shape[0]
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    v0, v1, v2 = coords[i0], coords[i1], coords[i2]

    def cotangent(a, b, c):
        ba = b - a
        ca = c - a
        cos = np.einsum('ij,ij->i', ba, ca)
        sin = np.linalg.norm(np.cross(ba, ca), axis=1)
        out = np.zeros_like(cos)
        m = sin > 1e-12
        out[m] = cos[m] / sin[m]
        return out

    cot0 = cotangent(v1, v0, v2)
    cot1 = cotangent(v2, v1, v0)
    cot2 = cotangent(v0, v2, v1)

    I = np.concatenate([i1, i2, i2, i0, i0, i1])
    J = np.concatenate([i2, i1, i0, i2, i1, i0])
    W_data = 0.5 * np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])

    W = sparse.coo_matrix((W_data, (I, J)), shape=(V, V)).tocsr()
    diag = np.array(-W.sum(axis=1)).ravel()
    L = sparse.diags(diag) + W
    M = _vertex_areas(coords, faces)
    return L.tocsr(), M


def compute_eigenmodes_from_mesh(coords: np.ndarray, faces: np.ndarray, K: int = 200):
    L, M = _cotangent_laplacian(coords, faces)
    M_sp = sparse.diags(M + 1e-12)
    K_req = min(K + 1, L.shape[0] - 2)
    try:
        lam, Phi = eigsh(L, k=K_req, M=M_sp, sigma=0.0, which='LM')
    except Exception:
        # Fallback to fewer modes if the large request is unstable
        K_req = max(10, min(60, L.shape[0] - 2))
        lam, Phi = eigsh(L, k=K_req, M=M_sp, sigma=0.0, which='LM')

    idx = np.argsort(lam)
    lam, Phi = lam[idx], Phi[:, idx]
    # drop DC
    lam = lam[1:]
    Phi = Phi[:, 1:]
    # keep strictly positive, finite eigenvalues
    mask = np.isfinite(lam) & (lam > 1e-12)
    lam = lam[mask]
    Phi = Phi[:, mask]
    return Phi, lam, M


def load_fsaverage_mesh(fs_level: str = "fsaverage5", hemi: str = "L") -> SurfMesh:
    fs = datasets.fetch_surf_fsaverage(fs_level)
    surf_path = fs.pial_left if hemi.upper() == "L" else fs.pial_right
    g = nib.load(surf_path)
    coords = g.darrays[0].data.astype(np.float64)
    faces = g.darrays[1].data.astype(np.int64)
    areas = _vertex_areas(coords, faces)
    return SurfMesh(coords=coords, faces=faces, areas=areas, hemi=hemi.upper())


def compute_eigenmodes_fsaverage(fs_level: str = "fsaverage5", hemi: str = "L", K: int = 200) -> Eigenmodes:
    mesh = load_fsaverage_mesh(fs_level, hemi)
    Phi, lam, M = compute_eigenmodes_from_mesh(mesh.coords, mesh.faces, K=K)
    return Eigenmodes(Phi=Phi, lam=lam, weights=mesh.areas, hemi=hemi)


# -------------------------------
# BOLD projection utilities
# -------------------------------

def sample_bold_to_surface(img_4d: nib.Nifti1Image, hemi: str = "L", fs_level: str = "fsaverage5") -> np.ndarray:
    """
    Sample a 4D BOLD volume onto an fsaverage surface using nilearn 0.12 API.
    Returns (n_vertices, n_timepoints). NaNs/inf -> 0.
    """
    fs = datasets.fetch_surf_fsaverage(fs_level)
    surf_mesh = fs.pial_left if hemi.upper() == "L" else fs.pial_right

    n_t = img_4d.shape[3]
    samples = []
    for t in range(n_t):
        img_3d = index_img(img_4d, t)
        vals = surface.vol_to_surf(img_3d, surf_mesh, radius=3.0)  # (n_vertices,)
        samples.append(vals)
    X = np.column_stack(samples)  # (V,T)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def project_timeseries_to_modes(Y: np.ndarray, eig: Eigenmodes) -> np.ndarray:
    Yc = Y - Y.mean(axis=1, keepdims=True)
    W = eig.weights[:, None]
    A = (Yc.T @ (eig.Phi * W))  # (T, V) @ (V, K)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    return A  # (T, K)


# -------------------------------
# TextGrid parsing & events stitching
# -------------------------------

def textgrid_to_events(textgrid_path: Path, tier_name: str = None) -> pd.DataFrame:
    """
    Parse a Praat TextGrid file into a DataFrame with columns:
      onset (s), duration (s), word, sentence_id
    Sentence increments whenever a token ends with [.!?]
    """
    try:
        import textgrid  # pip install textgrid
        tg = textgrid.TextGrid.fromFile(str(textgrid_path))
        if tier_name is None:
            cand = [t for t in tg.tiers if re.search(r'word|Word|token|Token', t.name)]
            tier = cand[0] if cand else tg.tiers[0]
        else:
            tier = next(t for t in tg.tiers if t.name == tier_name)
        rows = []
        sent_id = 0
        for itv in tier.intervals:
            word = (itv.mark or "").strip()
            onset = float(itv.minTime)
            dur = float(itv.maxTime - itv.minTime)
            if word:
                rows.append((onset, dur, word, sent_id))
                if re.search(r'[.!?]$', word):
                    sent_id += 1
        return pd.DataFrame(rows, columns=["onset", "duration", "word", "sentence_id"])
    except Exception:
        # minimal fallback parser
        txt = Path(textgrid_path).read_text(encoding="utf-8", errors="ignore")
        pat = re.compile(r'intervals \[\d+\]:\s*?xmin = ([\d\.]+)\s*?xmax = ([\d\.]+)\s*?text = "(.*?)"', re.S)
        rows = []
        sent_id = 0
        for xmin, xmax, mark in pat.findall(txt):
            word = (mark or "").strip()
            onset = float(xmin); dur = float(xmax) - float(xmin)
            if word:
                rows.append((onset, dur, word, sent_id))
                if re.search(r'[.!?]$', word):
                    sent_id += 1
        return pd.DataFrame(rows, columns=["onset", "duration", "word", "sentence_id"])


def load_lpp_sections(stim_dir: Path, lang: str = "EN") -> Dict[int, pd.DataFrame]:
    """
    Load all LPP sections for a language from a stimuli/annotation directory.
    Expects files like: lppEN_section1.TextGrid (or section01).
    Returns dict: section_number -> events DataFrame (onset,duration,word,sentence_id)
    """
    lang = lang.upper()
    pat = re.compile(rf"lpp{lang}_section0?(\d+)\.TextGrid$", re.IGNORECASE)
    out = {}
    for p in sorted(Path(stim_dir).glob(f"*lpp{lang}_section*.TextGrid")):
        m = pat.search(p.name)
        if not m:
            continue
        sec = int(m.group(1))
        df = textgrid_to_events(p)
        out[sec] = df
    if not out:
        raise FileNotFoundError(f"No TextGrid files found in {stim_dir} for language {lang}.")
    return out


def stitch_sections(sections: Dict[int, pd.DataFrame], order: List[int]) -> pd.DataFrame:
    """
    Concatenate specified sections in 'order' and add cumulative offsets so
    onsets are continuous. Returns a single events DataFrame.
    """
    frames = []
    t_offset = 0.0
    sent_offset = 0
    for sec in order:
        df = sections[sec].copy()
        df["onset"] = df["onset"] + t_offset
        df["sentence_id"] = df["sentence_id"] + sent_offset
        frames.append(df)
        # update offsets
        t_offset = df["onset"].iloc[-1] + df["duration"].iloc[-1]
        sent_offset = int(df["sentence_id"].iloc[-1]) + 1
    return pd.concat(frames, ignore_index=True)


def boundaries_from_events(events: pd.DataFrame) -> np.ndarray:
    m = events["word"].astype(str).str.endswith(('.', '!', '?'))
    return (events.loc[m, "onset"] + events.loc[m, "duration"]).to_numpy()


# -------------------------------
# ERME
# -------------------------------

def erme(Az: np.ndarray, boundaries_sec: np.ndarray, tr: float,
         k_idx: List[int], pre: float = 12.0, post: float = 24.0, step: float = 2.0):
    E_t = (Az[:, k_idx] ** 2).mean(axis=1)  # safe if k_idx non-empty
    T = Az.shape[0]
    lags = np.arange(-pre, post + 1e-6, step)
    lag_idx = np.round(lags / tr).astype(int)
    b_idx = np.round(boundaries_sec / tr).astype(int)
    b_idx = b_idx[(b_idx > 0) & (b_idx < T)]
    base_mask = (lags >= -12) & (lags <= -6)
    if not base_mask.any():
        base_mask = (lags < 0)
    erme_vals = np.zeros_like(lags, dtype=float)
    n_accum = 0
    for b in b_idx:
        idxs = b + lag_idx
        if idxs.min() < 0 or idxs.max() >= T:
            continue
        seg = E_t[idxs]
        baseline = seg[base_mask].mean() if base_mask.any() else 0.0
        erme_vals += (seg - baseline)
        n_accum += 1
    if n_accum > 0:
        erme_vals /= n_accum
    return lags, erme_vals


# -------------------------------
# High-level helpers
# -------------------------------

def sample_and_project(bold_path: Path, fs_level: str, K_modes: int):
    img = nib.load(str(bold_path))
    tr = img.header.get_zooms()[3]
    # surfaces
    Y_L = sample_bold_to_surface(img, hemi="L", fs_level=fs_level)
    Y_R = sample_bold_to_surface(img, hemi="R", fs_level=fs_level)
    # eigenmodes
    eigL = compute_eigenmodes_fsaverage(fs_level=fs_level, hemi="L", K=K_modes)
    eigR = compute_eigenmodes_fsaverage(fs_level=fs_level, hemi="R", K=K_modes)
    # project
    A_L = project_timeseries_to_modes(Y_L, eigL)
    A_R = project_timeseries_to_modes(Y_R, eigR)
    # z-score (nan-safe)
    A_Lz = zscore(A_L, axis=0, ddof=1)
    A_Rz = zscore(A_R, axis=0, ddof=1)
    A_Lz = np.nan_to_num(A_Lz, nan=0.0, posinf=0.0, neginf=0.0)
    A_Rz = np.nan_to_num(A_Rz, nan=0.0, posinf=0.0, neginf=0.0)
    # spectra (nan-safe)
    E_L = np.nanvar(A_Lz, axis=0)
    E_R = np.nanvar(A_Rz, axis=0)
    return {
        "TR": tr,
        "A_Lz": A_Lz, "A_Rz": A_Rz,
        "E_L": E_L, "E_R": E_R,
        "lam_L": eigL.lam, "lam_R": eigR.lam
    }


def run_lpp_textgrid_pipeline(
    bold_path: Path,
    stim_dir: Path,
    lang: str = "EN",
    fs_level: str = "fsaverage5",
    K_modes: int = 200,
    bands: Dict[str, Tuple[int, int]] = None,
    run_to_sections: Optional[Dict[int, List[int]]] = None,
):
    """
    Run pipeline for one bold file using TextGrids in stim_dir.
    """
    # Parse run number from filename
    m = re.search(r"run-(\d+)", Path(bold_path).name)
    if not m:
        raise ValueError(f"Cannot infer run number from {Path(bold_path).name}")
    run_no = int(m.group(1))

    # Load sections, decide order
    sections = load_lpp_sections(stim_dir, lang=lang)
    if run_to_sections is None:
        if run_no in sections:
            order = [run_no]
        elif (run_no % 100) in sections:
            order = [run_no % 100]
        else:
            raise ValueError(f"No section found matching run {run_no}. Provide run_to_sections mapping.")
    else:
        if run_no not in run_to_sections:
            raise ValueError(f"run_to_sections has no entry for run {run_no}")
        order = run_to_sections[run_no]

    events = stitch_sections(sections, order)
    boundaries = boundaries_from_events(events)

    # Sample + project
    proj = sample_and_project(bold_path, fs_level=fs_level, K_modes=K_modes)

    # How many modes did we actually get?
    K_modes_eff = int(proj["A_Lz"].shape[1])

    # If no/too few modes, return an empty result (runner will skip gracefully)
    if K_modes_eff < 3:
        return {
            "TR": proj["TR"],
            "lam": np.array([]),
            "E_mean": np.array([]),
            "lags": np.array([]),
            "ERME": {},
            "bands": {},
            "events": events,
        }

    # Default bands: safe thirds of available modes (guaranteed non-empty, ascending)
    if bands is None:
        a = max(1, K_modes_eff // 3)
        b = max(2, (2 * K_modes_eff) // 3)
        # clamp so we always have 0 < a < b < K
        a = min(a, K_modes_eff - 2)
        b = min(b, K_modes_eff - 1)
        bands = {"low": (0, a), "mid": (a, b), "high": (b, K_modes_eff)}

    # ERME per band (skip any degenerate ranges)
    erme_out = {}
    lags_out = None
    for name, (k0, k1) in bands.items():
        k0 = int(max(0, k0))
        k1 = int(min(K_modes_eff, k1))
        if k1 - k0 <= 0:
            continue
        idx = list(range(k0, k1))
        lags_out, erme_L = erme(proj["A_Lz"], boundaries, proj["TR"], idx)
        _,        erme_R = erme(proj["A_Rz"], boundaries, proj["TR"], idx)
        erme_out[name] = 0.5 * (erme_L + erme_R)

    # Spectrum and eigenvalues (filter invalid downstream in the runner)
    E_mean = 0.5 * (proj["E_L"] + proj["E_R"])
    lam = 0.5 * (proj["lam_L"] + proj["lam_R"])

    return {
        "TR": proj["TR"],
        "lam": lam,
        "E_mean": E_mean,
        "lags": lags_out if lags_out is not None else np.array([]),
        "ERME": erme_out,
        "bands": bands,
        "events": events
    }