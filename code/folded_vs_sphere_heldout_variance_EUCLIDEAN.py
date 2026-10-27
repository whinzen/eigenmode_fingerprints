from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from lapy import Solver, TriaMesh

BASE = Path.cwd()
OUT = BASE / "pang_out/geometry_specificity/heldout_variance_euclidean"
OUT.mkdir(parents=True, exist_ok=True)

CACHE = (
    BASE / "pang_out/nuisance_robustness/"
    "vertex_space_linguistic_encoding/corrected_vertex_cache"
)
MANIFEST = (
    BASE / "pang_out/nuisance_robustness/"
    "vertex_space_linguistic_encoding/"
    "structured_null_10000/vertex_cache_manifest.csv"
)
MASK_ROOT = BASE / "pang_out/geometry_specificity/crossvalidated_pca"
SPHERE_ROOT = BASE / "pang_out/geometry_specificity/sphere_control"
EIG_ROOT = BASE / "data/template_eigenmodes/fsaverage5"
SURF_ROOT = BASE / "pang_surfaces/fsaverage5"

KS = [3, 8, 15, 24, 35, 48]

def prepare_basis(phi, M, valid, max_k):
    """
    Remove constant component and M-orthonormalize basis
    on the common valid-vertex domain.
    """
    Mv = M[valid][:, valid].tocsr()
    one = np.ones(valid.sum())
    mone = Mv @ one
    denom = one @ mone

    P = phi[valid, 1:max_k + 1].astype(np.float64)
    P -= np.outer(one, (mone @ P) / denom)

    G = P.T @ (Mv @ P)
    L = cholesky(G, lower=True, check_finite=True)

    # Q = P @ inv(L.T), so Q.T M Q = I.
    Q = solve_triangular(L, P.T, lower=True).T

    gram_error = np.max(
        np.abs(Q.T @ (Mv @ Q) - np.eye(max_k))
    )
    assert gram_error < 1e-7, gram_error

    return Q, Mv, mone, denom

def evaluate_run(Z, Q, Mv, mone, denom):
    """
    Returns cumulative projected variance and total
    mass-weighted spatial variance.
    """
    Y = np.asarray(Z, dtype=np.float64)

    # Remove spatial mean at every time point.
    Y -= np.outer(np.ones(Y.shape[0]), (mone @ Y) / denom)

    MY = Mv @ Y
    total = np.sum(Y * MY)

    coefficients = Q.T @ MY
    cumulative = np.cumsum(
        np.sum(coefficients ** 2, axis=1)
    )

    return cumulative, total

manifest = pd.read_csv(MANIFEST)
assert len(manifest) == 439
assert manifest.subject.nunique() == 49

bases = {}

for hemi in ["L", "R"]:
    print("Preparing", hemi, flush=True)

    valid = np.load(
        MASK_ROOT / f"common_valid_mask_{hemi}.npy"
    ).astype(bool)

    surface = TriaMesh.read_vtk(
        str(SURF_ROOT / f"{'lh' if hemi == 'L' else 'rh'}.white.vtk")
    )
    from scipy.sparse import identity
    M = identity(10242, format="csr", dtype=np.float64)

    folded = np.load(
        EIG_ROOT / f"phi_{hemi}.npy"
    )
    sphere = np.load(
        SPHERE_ROOT / f"sphere_eigenmodes_{hemi}.npy"
    )

    Qf, Mv, mone, denom = prepare_basis(
        folded, M, valid, max(KS)
    )
    Qs, _, _, _ = prepare_basis(
        sphere, M, valid, max(KS)
    )

    bases[hemi] = {
        "valid": valid,
        "Qf": Qf,
        "Qs": Qs,
        "Mv": Mv,
        "mone": mone,
        "denom": denom,
    }

rows = []

for idx, row in enumerate(manifest.itertuples(index=False), 1):
    subject = str(row.subject)
    run = int(row.run)

    for hemi in ["L", "R"]:
        b = bases[hemi]

        path = (
            CACHE / subject / f"run-{run}" / f"Z_{hemi}.npy"
        )

        Z = np.load(path, mmap_mode="r")
        Z = np.asarray(Z[b["valid"], :], dtype=np.float64)

        cum_f, total_f = evaluate_run(
            Z.copy(), b["Qf"], b["Mv"],
            b["mone"], b["denom"]
        )
        cum_s, total_s = evaluate_run(
            Z.copy(), b["Qs"], b["Mv"],
            b["mone"], b["denom"]
        )

        assert np.isclose(total_f, total_s, rtol=1e-10)

        for K in KS:
            rf = cum_f[K - 1] / total_f
            rs = cum_s[K - 1] / total_s

            rows.append({
                "subject": subject,
                "run": run,
                "hemisphere": hemi,
                "K": K,
                "folded_variance": rf,
                "sphere_variance": rs,
                "delta": rf - rs,
            })

    if idx % 25 == 0:
        print(f"Processed {idx}/439 runs", flush=True)

run_df = pd.DataFrame(rows)
run_df.to_csv(OUT / "run_level_variance.csv", index=False)

# Equal weighting of runs within hemisphere, then hemispheres.
hemi_df = (
    run_df.groupby(["subject", "hemisphere", "K"], as_index=False)
    [["folded_variance", "sphere_variance", "delta"]]
    .mean()
)
hemi_df.to_csv(
    OUT / "participant_hemisphere_variance.csv",
    index=False
)

subject_df = (
    hemi_df.groupby(["subject", "K"], as_index=False)
    [["folded_variance", "sphere_variance", "delta"]]
    .mean()
)
subject_df.to_csv(
    OUT / "participant_bilateral_variance.csv",
    index=False
)

# Paired participant bootstrap and sign-flip test.
rng = np.random.default_rng(20261021)
summary = []

for K, df in subject_df.groupby("K"):
    d = df.delta.to_numpy()
    n = len(d)

    B = 20000
    bootstrap = np.mean(
        d[rng.integers(0, n, size=(B, n))],
        axis=1
    )

    signs = rng.choice(
        [-1, 1], size=(B, n)
    )
    perm = np.mean(signs * d, axis=1)

    observed = np.mean(d)

    p_two = (
        1 + np.sum(np.abs(perm) >= abs(observed))
    ) / (B + 1)

    lo, hi = np.percentile(bootstrap, [2.5, 97.5])

    summary.append({
        "K": K,
        "n_participants": n,
        "folded_mean": df.folded_variance.mean(),
        "sphere_mean": df.sphere_variance.mean(),
        "delta_mean": observed,
        "delta_pp": 100 * observed,
        "ci_low_pp": 100 * lo,
        "ci_high_pp": 100 * hi,
        "p_two_sided": p_two,
        "n_positive": int(np.sum(d > 0)),
    })

summary = pd.DataFrame(summary).sort_values("K")
summary.to_csv(OUT / "variance_capture_summary.csv", index=False)

print("\nFINAL RESULTS")
print(summary.to_string(index=False))
print("\nSaved to:", OUT)
