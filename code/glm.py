"""
glm.py

OLS regression utilities for eigenmode energy time series.

Expected usage:
    run_ols_and_save(
        func_file: Path to bold.func.gii,
        X: np.ndarray [T] or [T x P],
        out_path: Path to output .npy,
        run_id: int,
        predictor_names: list[str]
    )
"""

import numpy as np
import nibabel as nib
from pathlib import Path

# ----------------------------
# Core OLS
# ----------------------------
def ols_fit(Y, X):
    """
    Ordinary Least Squares:
        Y: [T x K]  (time x modes)
        X: [T x P]  (design matrix)
    Returns:
        beta: [P x K]
    """
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X.T @ Y
    return beta


# ----------------------------
# Load surface time series
# ----------------------------
def load_func_gii(func_file):
    """
    Load fsaverage5 func.gii and return:
        Y: [T x K] eigenmode energy time series
    """
    img = nib.load(str(func_file))

    # func.gii: darrays = timepoints, each [vertices]
    data = np.array([da.data for da in img.darrays])  # [T x V]
    return data


# ----------------------------
# Public API
# ----------------------------
def run_ols_and_save(
    func_file,
    X,
    out_path,
    run_id=None,
    predictor_names=None,
):
    """
    Run OLS and save beta coefficients.

    Parameters
    ----------
    func_file : Path or str
        Path to *_bold.func.gii
    X : np.ndarray
        Design matrix [T] or [T x P]
    out_path : Path
        Where to save .npy
    run_id : int
        Run identifier (for bookkeeping)
    predictor_names : list[str]
        Names of predictors
    """

    # ----------------------------
    # Safety checks
    # ----------------------------
    if isinstance(X, Path):
        raise TypeError(
            f"X is a Path ({X}) — did you forget np.load(regressor)?"
        )

    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be np.ndarray, got {type(X)}")

    X = np.asarray(X)

    if X.ndim == 1:
        X = X[:, None]  # [T x 1]

    T = X.shape[0]

    # ----------------------------
    # Load data
    # ----------------------------
    Y = load_func_gii(func_file)  # [T x V]

    if Y.shape[0] != T:
        raise ValueError(
            f"Length mismatch: X has {T} TRs but Y has {Y.shape[0]}"
        )

    # ----------------------------
    # Add intercept
    # ----------------------------
    intercept = np.ones((T, 1))
    X_design = np.hstack([intercept, X])

    if predictor_names is None:
        predictor_names = ["predictor"]

    names = ["intercept"] + list(predictor_names)

    # ----------------------------
    # Fit
    # ----------------------------
    beta = ols_fit(Y, X_design)  # [P+1 x V]

    # ----------------------------
    # Save
    # ----------------------------
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(
        out_path,
        {
            "beta": beta,
            "predictors": names,
            "run_id": run_id,
            "func_file": str(func_file),
        },
        allow_pickle=True,
    )

    print(f"✅ GLM written: {out_path}")