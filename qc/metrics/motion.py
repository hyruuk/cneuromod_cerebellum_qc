"""
Motion metrics derived from fMRIPrep confounds timeseries.

Computes FD, DVARS, scrubbing statistics, and session-level motion trends.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress


def load_confounds(
    confounds_path: Path,
    confounds_json_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Load fMRIPrep confounds TSV and its JSON sidecar.

    Parameters
    ----------
    confounds_path:
        Path to *_desc-confounds_timeseries.tsv
    confounds_json_path:
        Path to *_desc-confounds_timeseries.json (optional metadata)

    Returns
    -------
    df:
        DataFrame with one row per TR.
    meta:
        Dict from JSON sidecar (empty dict if unavailable).
    """
    df = pd.read_csv(confounds_path, sep="\t", low_memory=False)

    meta = {}
    if confounds_json_path and Path(confounds_json_path).exists():
        with open(confounds_json_path) as f:
            meta = json.load(f)

    return df, meta


def compute_motion_metrics(
    df: pd.DataFrame,
    fd_threshold: float = 0.5,
    dvars_threshold: float = 1.5,
) -> Dict[str, float]:
    """
    Compute motion-related QC metrics from a confounds DataFrame.

    Parameters
    ----------
    df:
        Confounds timeseries DataFrame.
    fd_threshold:
        Framewise displacement threshold for scrubbing (mm).
    dvars_threshold:
        Standardized DVARS threshold for scrubbing.

    Returns
    -------
    Dict with motion metrics.
    """
    result: Dict[str, float] = {}
    n_trs = len(df)
    result["n_trs"] = float(n_trs)

    # --- Framewise Displacement ---
    if "framewise_displacement" in df.columns:
        fd = df["framewise_displacement"].values.copy().astype(float)
        # FD[0] is NaN in fMRIPrep output (no previous frame)
        fd_valid = fd[np.isfinite(fd)]
        result["mean_fd"] = float(np.mean(fd_valid)) if len(fd_valid) > 0 else float("nan")
        result["median_fd"] = float(np.median(fd_valid)) if len(fd_valid) > 0 else float("nan")
        result["max_fd"] = float(np.max(fd_valid)) if len(fd_valid) > 0 else float("nan")
        result["pct_fd_above_threshold"] = (
            float(100.0 * np.sum(fd_valid > fd_threshold) / len(fd_valid))
            if len(fd_valid) > 0 else float("nan")
        )
        result["n_fd_above_threshold"] = float(np.sum(fd_valid > fd_threshold))
    else:
        for k in ["mean_fd", "median_fd", "max_fd", "pct_fd_above_threshold", "n_fd_above_threshold"]:
            result[k] = float("nan")
        fd = np.full(n_trs, np.nan)

    # --- DVARS ---
    if "std_dvars" in df.columns:
        dvars = df["std_dvars"].values.copy().astype(float)
        dvars_valid = dvars[np.isfinite(dvars)]
        result["mean_dvars"] = float(np.mean(dvars_valid)) if len(dvars_valid) > 0 else float("nan")
        result["max_dvars"] = float(np.max(dvars_valid)) if len(dvars_valid) > 0 else float("nan")
        result["pct_dvars_above_threshold"] = (
            float(100.0 * np.sum(dvars_valid > dvars_threshold) / len(dvars_valid))
            if len(dvars_valid) > 0 else float("nan")
        )
    elif "dvars" in df.columns:
        dvars = df["dvars"].values.copy().astype(float)
        dvars_valid = dvars[np.isfinite(dvars)]
        result["mean_dvars"] = float(np.mean(dvars_valid)) if len(dvars_valid) > 0 else float("nan")
        result["max_dvars"] = float(np.max(dvars_valid)) if len(dvars_valid) > 0 else float("nan")
        result["pct_dvars_above_threshold"] = float("nan")
        dvars = np.full(n_trs, np.nan)
    else:
        for k in ["mean_dvars", "max_dvars", "pct_dvars_above_threshold"]:
            result[k] = float("nan")
        dvars = np.full(n_trs, np.nan)

    # --- RMSD ---
    if "rmsd" in df.columns:
        rmsd = df["rmsd"].values.astype(float)
        rmsd_valid = rmsd[np.isfinite(rmsd)]
        result["mean_rmsd"] = float(np.mean(rmsd_valid)) if len(rmsd_valid) > 0 else float("nan")
    else:
        result["mean_rmsd"] = float("nan")

    # --- Scrubbing (combined criterion) ---
    fd_arr = df["framewise_displacement"].values.astype(float) if "framewise_displacement" in df.columns else np.full(n_trs, np.nan)
    std_dvars_arr = df["std_dvars"].values.astype(float) if "std_dvars" in df.columns else np.full(n_trs, np.nan)

    # First TR is always NaN for FD — treat as not scrubbed
    fd_bad = np.nan_to_num(fd_arr, nan=0.0) > fd_threshold
    dvars_bad = np.nan_to_num(std_dvars_arr, nan=0.0) > dvars_threshold
    scrub_mask = fd_bad | dvars_bad

    result["n_scrubbed"] = float(np.sum(scrub_mask))
    result["n_usable"] = float(n_trs - np.sum(scrub_mask))
    result["pct_usable"] = float(100.0 * result["n_usable"] / n_trs) if n_trs > 0 else float("nan")

    # --- fMRIPrep spike regressors ---
    outlier_cols = [c for c in df.columns if re.match(r"motion_outlier\d+", c)]
    result["n_motion_outlier_regressors"] = float(len(outlier_cols))

    return result


def compute_session_fd_trend(
    session_labels: List[str],
    session_mean_fds: List[float],
) -> Dict[str, float]:
    """
    Fit a linear regression of mean FD across sessions to detect trends.

    Positive slope → motion increases across the study (fatigue / habituation).

    Parameters
    ----------
    session_labels:
        List of session label strings (e.g., ['ses-001', 'ses-002', ...]).
        Used to extract numeric session indices for regression.
    session_mean_fds:
        Mean FD per session (same order as session_labels).

    Returns
    -------
    Dict with keys: slope, intercept, r_squared, p_value, n_sessions.
    """
    result: Dict[str, float] = {
        "fd_trend_slope": float("nan"),
        "fd_trend_r2": float("nan"),
        "fd_trend_pval": float("nan"),
        "fd_trend_n": float("nan"),
    }

    # Extract numeric session index from label (e.g., 'ses-007' → 7)
    indices = []
    fds = []
    for label, fd in zip(session_labels, session_mean_fds):
        m = re.search(r"ses-(\d+)", label)
        if m and np.isfinite(fd):
            indices.append(int(m.group(1)))
            fds.append(fd)

    if len(indices) < 3:
        return result

    x = np.array(indices, dtype=float)
    y = np.array(fds, dtype=float)

    try:
        slope, intercept, r, p, _ = linregress(x, y)
        result["fd_trend_slope"] = float(slope)
        result["fd_trend_intercept"] = float(intercept)
        result["fd_trend_r2"] = float(r ** 2)
        result["fd_trend_pval"] = float(p)
        result["fd_trend_n"] = float(len(indices))
    except Exception:
        pass

    return result


def parse_acompcor_stats(
    meta: dict,
    mask_types: Tuple[str, ...] = ("CSF", "WM", "combined"),
) -> Dict[str, float]:
    """
    Parse aCompCor variance statistics from fMRIPrep confounds JSON metadata.

    fMRIPrep 20.2.5 retains ~370 aCompCor components (all above noise floor) per
    mask type. This function extracts:
    - n_retained: total retained components
    - cumvar_final: cumulative variance explained by all retained components
    - n_for_50pct: components needed to reach 50% cumulative variance
    - n_for_80pct: components needed to reach 80% cumulative variance
    - cumvar_curve: full curve as list (for plotting)

    Parameters
    ----------
    meta:
        Dict loaded from *_desc-confounds_timeseries.json.
    mask_types:
        Which mask types to process.

    Returns
    -------
    Flat dict with keys like 'acompcor_CSF_n_retained', etc.
    """
    result: Dict[str, float] = {}

    for mask_type in mask_types:
        retained = {
            k: v
            for k, v in meta.items()
            if k.startswith("a_comp_cor")
            and isinstance(v, dict)
            and v.get("Retained", False)
            and v.get("Mask") == mask_type
        }

        prefix = f"acompcor_{mask_type}"

        if not retained:
            result[f"{prefix}_n_retained"] = float("nan")
            result[f"{prefix}_cumvar_final"] = float("nan")
            result[f"{prefix}_n_for_50pct"] = float("nan")
            result[f"{prefix}_n_for_80pct"] = float("nan")
            continue

        # Sort by numeric component index (not lexicographic) so cumvar curve is monotonic
        def _comp_index(key: str) -> int:
            m = re.search(r"(\d+)$", key)
            return int(m.group(1)) if m else 0

        sorted_items = sorted(retained.items(), key=lambda kv: _comp_index(kv[0]))
        cumvar_curve = [v["CumulativeVarianceExplained"] for _, v in sorted_items]

        n_retained = len(cumvar_curve)
        cumvar_final = cumvar_curve[-1]

        # Components needed to reach 50% / 80% variance.
        # Return NaN when the threshold is never reached (all retained components
        # together don't explain enough variance) so the plot can distinguish
        # "not reached" from "reached at the last component".
        n_for_50_raw = next((i + 1 for i, cv in enumerate(cumvar_curve) if cv >= 0.5), None)
        n_for_80_raw = next((i + 1 for i, cv in enumerate(cumvar_curve) if cv >= 0.8), None)
        n_for_50 = n_for_50_raw if n_for_50_raw is not None else float("nan")
        n_for_80 = n_for_80_raw if n_for_80_raw is not None else float("nan")

        result[f"{prefix}_n_retained"] = float(n_retained)
        result[f"{prefix}_cumvar_final"] = float(cumvar_final)
        result[f"{prefix}_n_for_50pct"] = n_for_50  # NaN if 50% never reached
        result[f"{prefix}_n_for_80pct"] = n_for_80  # NaN if 80% never reached

    return result


def get_fd_series(df: pd.DataFrame) -> np.ndarray:
    """Return the framewise displacement time series as float array."""
    if "framewise_displacement" in df.columns:
        return df["framewise_displacement"].values.astype(float)
    return np.full(len(df), np.nan)


def get_motion_params(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Return 6-parameter motion array (trans_x, y, z, rot_x, y, z) if available.

    Returns
    -------
    Array of shape (n_trs, 6) or None if motion parameters are not present.
    """
    param_names = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    available = [c for c in param_names if c in df.columns]
    if len(available) < 6:
        return None
    return df[param_names].values.astype(float)
