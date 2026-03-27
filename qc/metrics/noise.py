"""
Noise and physiological contamination metrics.

Includes:
- 4th ventricle CSF signal correlation with cerebellar signal
- Temporal autocorrelation (AR1) of cerebellar mean time series
- Inter-lobule correlation matrix
- Cerebellum–motor cortex baseline correlation
- Carpet plot data extraction (cerebellar voxels × time)
- Session-level signal drift
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr


# aseg label IDs
_ASEG_4TH_VENT = 15
_ASEG_3RD_VENT = 14
_ASEG_L_CEREB_CORTEX = 8
_ASEG_R_CEREB_CORTEX = 47
_ASEG_L_LAT_VENT = 4
_ASEG_R_LAT_VENT = 43

# aparcaseg labels for precentral gyrus (motor cortex, Desikan-Killiany atlas)
_ASEG_L_PRECENTRAL = 1024
_ASEG_R_PRECENTRAL = 2024


def compute_csf_correlation(
    bold_data: np.ndarray,
    aseg_data: np.ndarray,
    mask_data: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute Pearson correlation between 4th ventricle CSF signal and cerebellar cortex.

    High correlation (r > 0.3) suggests CSF contamination from 4th ventricle
    due to partial-volume effects at 2mm resolution.

    Parameters
    ----------
    bold_data:
        4D float32 array (x, y, z, t).
    aseg_data:
        3D integer aseg array.
    mask_data:
        Optional brain mask.

    Returns
    -------
    Dict with keys:
        - 'csf_cereb_corr_L': Pearson r between 4th vent and L cerebellar cortex
        - 'csf_cereb_corr_R': Pearson r, right side
        - 'csf_4thvent_n_voxels': number of 4th-vent voxels surviving masking
        - 'csf_corr_warning': 1.0 if either correlation > 0.3, else 0.0
    """
    result: Dict[str, float] = {
        "csf_cereb_corr_L": float("nan"),
        "csf_cereb_corr_R": float("nan"),
        "csf_4thvent_n_voxels": 0.0,
        "csf_corr_warning": 0.0,
    }

    mask_bool = mask_data.astype(bool) if mask_data is not None else np.ones(bold_data.shape[:3], bool)

    # 4th ventricle ROI — may be excluded by brain mask (it's CSF)
    vent_mask = (aseg_data == _ASEG_4TH_VENT)
    # Don't restrict 4th vent to brain mask (it may be excluded)
    n_vent = vent_mask.sum()
    result["csf_4thvent_n_voxels"] = float(n_vent)

    if n_vent < 3:
        # Fall back to lateral ventricles if 4th vent is too small
        vent_mask = np.isin(aseg_data, [_ASEG_L_LAT_VENT, _ASEG_R_LAT_VENT]) & mask_bool
        n_vent = vent_mask.sum()
        result["csf_4thvent_n_voxels"] = float(n_vent)
        if n_vent < 3:
            return result

    vent_ts = bold_data[vent_mask, :].mean(axis=0)  # shape (n_trs,)

    for side, label_id, key in [
        ("L", _ASEG_L_CEREB_CORTEX, "csf_cereb_corr_L"),
        ("R", _ASEG_R_CEREB_CORTEX, "csf_cereb_corr_R"),
    ]:
        cereb_mask = (aseg_data == label_id) & mask_bool
        if cereb_mask.sum() < 3:
            continue
        cereb_ts = bold_data[cereb_mask, :].mean(axis=0)
        if np.std(vent_ts) < 1e-6 or np.std(cereb_ts) < 1e-6:
            continue
        r, _ = pearsonr(vent_ts, cereb_ts)
        result[key] = float(r)

    # Warning flag
    l_corr = result["csf_cereb_corr_L"]
    r_corr = result["csf_cereb_corr_R"]
    if (np.isfinite(l_corr) and l_corr > 0.3) or (np.isfinite(r_corr) and r_corr > 0.3):
        result["csf_corr_warning"] = 1.0

    return result


def compute_ar1(
    bold_data: np.ndarray,
    aseg_data: np.ndarray,
    mask_data: Optional[np.ndarray] = None,
    motion_params: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute AR(1) coefficient of mean cerebellar time series.

    Optionally regress out 6 motion parameters before computing AR(1)
    to get a purer estimate of intrinsic autocorrelation.

    Parameters
    ----------
    bold_data:
        4D float32 array.
    aseg_data:
        3D aseg array.
    mask_data:
        Optional brain mask.
    motion_params:
        Optional (n_trs, 6) motion parameter array. If provided, regressed out.

    Returns
    -------
    Dict with keys:
        - 'ar1_L_cereb': AR(1) for left cerebellar cortex mean time series
        - 'ar1_R_cereb': AR(1) for right cerebellar cortex
        - 'ar1_cereb_mean': average of left and right
    """
    result: Dict[str, float] = {
        "ar1_L_cereb": float("nan"),
        "ar1_R_cereb": float("nan"),
        "ar1_cereb_mean": float("nan"),
    }

    mask_bool = mask_data.astype(bool) if mask_data is not None else np.ones(bold_data.shape[:3], bool)

    for key, label_id in [("ar1_L_cereb", _ASEG_L_CEREB_CORTEX), ("ar1_R_cereb", _ASEG_R_CEREB_CORTEX)]:
        roi_mask = (aseg_data == label_id) & mask_bool
        if roi_mask.sum() < 3:
            continue
        ts = bold_data[roi_mask, :].mean(axis=0).astype(float)

        # Optional motion regression
        if motion_params is not None and motion_params.shape[0] == len(ts):
            params = motion_params.copy()
            # Replace NaN motion params with 0
            params = np.nan_to_num(params, nan=0.0)
            # Demean ts and params
            ts = ts - ts.mean()
            params = params - params.mean(axis=0, keepdims=True)
            # Least-squares regression
            try:
                coeffs, _, _, _ = np.linalg.lstsq(params, ts, rcond=None)
                ts = ts - params @ coeffs
            except np.linalg.LinAlgError:
                pass

        # Demean
        ts = ts - ts.mean()
        if len(ts) < 3 or np.std(ts) < 1e-6:
            continue

        # AR(1): correlation between ts[1:] and ts[:-1]
        r, _ = pearsonr(ts[:-1], ts[1:])
        result[key] = float(r)

    l = result["ar1_L_cereb"]
    r = result["ar1_R_cereb"]
    if np.isfinite(l) and np.isfinite(r):
        result["ar1_cereb_mean"] = float(0.5 * (l + r))
    elif np.isfinite(l):
        result["ar1_cereb_mean"] = l
    elif np.isfinite(r):
        result["ar1_cereb_mean"] = r

    return result


def compute_motor_cereb_correlation(
    bold_data: np.ndarray,
    aseg_data: np.ndarray,
    mask_data: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute baseline Pearson correlation between motor cortex and cerebellar time series.

    Uses precentral gyrus labels from aparcaseg (if present in aseg_data).
    High r (> 0.5) before denoising suggests shared variance that complicates
    cerebellar-specific interpretation.

    Note: This requires the aparcaseg labels (1024/2024) to be present in the
    aseg_dseg file. If not, falls back to a small sphere ROI approach using
    label 17 (Left Hippocampus) as a rough check — or simply returns NaN.
    """
    result: Dict[str, float] = {
        "motor_cereb_corr": float("nan"),
        "motor_cereb_corr_warning": 0.0,
    }

    mask_bool = mask_data.astype(bool) if mask_data is not None else np.ones(bold_data.shape[:3], bool)

    # Motor cortex: try aparcaseg precentral labels
    motor_mask = np.isin(aseg_data, [_ASEG_L_PRECENTRAL, _ASEG_R_PRECENTRAL]) & mask_bool
    if motor_mask.sum() < 5:
        # Not available in basic aseg — return NaN gracefully
        return result

    motor_ts = bold_data[motor_mask, :].mean(axis=0)

    # Cerebellar cortex
    cereb_mask = np.isin(aseg_data, [_ASEG_L_CEREB_CORTEX, _ASEG_R_CEREB_CORTEX]) & mask_bool
    if cereb_mask.sum() < 5:
        return result

    cereb_ts = bold_data[cereb_mask, :].mean(axis=0)

    if np.std(motor_ts) < 1e-6 or np.std(cereb_ts) < 1e-6:
        return result

    r, _ = pearsonr(motor_ts, cereb_ts)
    result["motor_cereb_corr"] = float(r)
    if np.isfinite(r) and abs(r) > 0.5:
        result["motor_cereb_corr_warning"] = 1.0

    return result


def compute_signal_drift(
    bold_data: np.ndarray,
    aseg_data: np.ndarray,
    mask_data: Optional[np.ndarray] = None,
    tr: float = 1.49,
) -> Dict[str, float]:
    """
    Compute linear drift slope in mean cerebellar signal over time.

    A large drift (>2% of mean signal) may indicate scanner instability or
    physiological drift. fMRIPrep applies cosine drifts as confounds, but the
    raw drift magnitude is still informative.

    Returns
    -------
    Dict with keys:
        - 'signal_drift_pct': linear slope as % of mean signal per minute
        - 'signal_drift_abs': slope in signal units / TR
    """
    result: Dict[str, float] = {
        "signal_drift_pct": float("nan"),
        "signal_drift_abs": float("nan"),
    }

    mask_bool = mask_data.astype(bool) if mask_data is not None else np.ones(bold_data.shape[:3], bool)

    cereb_mask = np.isin(aseg_data, [_ASEG_L_CEREB_CORTEX, _ASEG_R_CEREB_CORTEX]) & mask_bool
    if cereb_mask.sum() < 5:
        return result

    ts = bold_data[cereb_mask, :].mean(axis=0).astype(float)
    n = len(ts)
    if n < 10:
        return result

    mean_signal = float(np.mean(ts))
    if mean_signal <= 0:
        return result

    # Fit linear trend: y = slope * t + intercept
    t = np.arange(n, dtype=float) * tr  # time in seconds
    coeffs = np.polyfit(t, ts, 1)
    slope = coeffs[0]  # signal units / second

    # Convert to % per minute for interpretability
    slope_pct_per_min = float(100.0 * slope * 60.0 / mean_signal)

    result["signal_drift_pct"] = slope_pct_per_min
    result["signal_drift_abs"] = float(slope)

    return result


def extract_carpet_data(
    bold_data: np.ndarray,
    suit_data: np.ndarray,
    mask_data: Optional[np.ndarray] = None,
    n_voxels_max: int = 500,
    seed: int = 42,
) -> Dict:
    """
    Extract subsampled cerebellar voxel time series for carpet plot visualization.

    Voxels are randomly subsampled (for HTML size) and sorted by SUIT lobule label
    for visual organization.

    Parameters
    ----------
    bold_data:
        4D float32 array (x, y, z, t).
    suit_data:
        3D SUIT atlas array.
    mask_data:
        Optional brain mask.
    n_voxels_max:
        Maximum number of voxels to include in the carpet.
    seed:
        Random seed for reproducible subsampling.

    Returns
    -------
    Dict with keys:
        - 'data': 2D float32 array (n_voxels, n_trs) of z-scored time series
        - 'voxel_labels': list of SUIT lobule name per voxel (for y-axis ticks)
        - 'n_voxels': actual number of voxels
    """
    from qc.atlas import SUIT_LABEL_MAP

    cereb_mask = (suit_data > 0)
    if mask_data is not None:
        cereb_mask = cereb_mask & mask_data.astype(bool)

    voxel_coords = np.argwhere(cereb_mask)
    if len(voxel_coords) == 0:
        return {"data": np.zeros((0, bold_data.shape[3])), "voxel_labels": [], "n_voxels": 0}

    # Sort by SUIT label for visual grouping, then subsample
    voxel_labels_raw = suit_data[cereb_mask]
    sort_idx = np.argsort(voxel_labels_raw, kind="stable")
    voxel_coords = voxel_coords[sort_idx]
    voxel_labels_raw = voxel_labels_raw[sort_idx]

    if len(voxel_coords) > n_voxels_max:
        rng = np.random.default_rng(seed)
        # Stratified subsample: pick proportionally from each lobule
        keep_idx = _stratified_subsample(voxel_labels_raw, n_voxels_max, rng)
        voxel_coords = voxel_coords[keep_idx]
        voxel_labels_raw = voxel_labels_raw[keep_idx]

    # Extract time series: shape (n_voxels, n_trs)
    ts = bold_data[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2], :]
    ts = ts.astype(np.float32)

    # Z-score each voxel row (over time)
    mean_ts = ts.mean(axis=1, keepdims=True)
    std_ts = ts.std(axis=1, keepdims=True)
    std_ts = np.where(std_ts < 1e-6, 1.0, std_ts)
    ts_z = (ts - mean_ts) / std_ts

    # Clip to ±3 for color scaling
    ts_z = np.clip(ts_z, -3, 3)

    # Map label IDs → names
    voxel_label_names = [SUIT_LABEL_MAP.get(int(lid), f"label_{lid}") for lid in voxel_labels_raw]

    return {
        "data": ts_z,
        "voxel_labels": voxel_label_names,
        "n_voxels": int(len(voxel_coords)),
    }


def _stratified_subsample(
    labels: np.ndarray,
    n_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Subsample indices proportionally across label groups."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_per_label = np.maximum(1, np.round(counts / counts.sum() * n_total).astype(int))
    # Adjust to exactly n_total
    diff = n_total - n_per_label.sum()
    n_per_label[0] += diff

    all_idx = []
    for label, n in zip(unique_labels, n_per_label):
        label_idx = np.where(labels == label)[0]
        n_pick = min(n, len(label_idx))
        picked = rng.choice(label_idx, n_pick, replace=False)
        all_idx.append(picked)

    combined = np.concatenate(all_idx)
    # Re-sort by label for visual grouping
    sort_order = np.argsort(labels[combined], kind="stable")
    return combined[sort_order]


def compute_interlobule_correlation(
    bold_data: np.ndarray,
    suit_data: np.ndarray,
    mask_data: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute Pearson correlation matrix between all SUIT lobule mean time series.

    High inter-lobule correlations (> 0.5) before denoising suggest poor spatial
    specificity — denoising will be critical for distinguishing lobular activity.

    Returns
    -------
    corr_matrix:
        (28, 28) float32 correlation matrix.
    lobule_names:
        List of lobule names corresponding to matrix rows/columns.
    """
    from qc.atlas import SUIT_LABEL_MAP, compute_lobule_timeseries

    lobule_ts_dict = compute_lobule_timeseries(bold_data, suit_data, SUIT_LABEL_MAP, mask_data)
    lobule_names = list(SUIT_LABEL_MAP.values())

    n = len(lobule_names)
    corr_matrix = np.full((n, n), np.nan, dtype=np.float32)

    ts_list = [lobule_ts_dict[name] for name in lobule_names]

    for i in range(n):
        for j in range(i, n):
            ts_i = ts_list[i]
            ts_j = ts_list[j]
            if np.isfinite(ts_i).all() and np.isfinite(ts_j).all():
                if np.std(ts_i) > 1e-6 and np.std(ts_j) > 1e-6:
                    r, _ = pearsonr(ts_i, ts_j)
                    corr_matrix[i, j] = r
                    corr_matrix[j, i] = r
            if i == j:
                corr_matrix[i, j] = 1.0

    return corr_matrix, lobule_names
