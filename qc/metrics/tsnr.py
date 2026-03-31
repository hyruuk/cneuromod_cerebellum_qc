"""
Temporal SNR (tSNR) computation and ROI extraction.

tSNR = mean(timeseries) / std(timeseries)

Computed voxelwise, then summarized per SUIT lobule and aseg ROI.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np

from qc.atlas import extract_roi_stats, SUIT_LABEL_MAP, ASEG_LABEL_MAP


def compute_tsnr_map(
    bold_img: nib.Nifti1Image,
    mask_img: Optional[nib.Nifti1Image] = None,
) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Compute voxelwise temporal SNR map from a 4D BOLD image.

    Parameters
    ----------
    bold_img:
        4D NIfTI image. Expected dtype is int32 (from fMRIPrep); cast to float32.
    mask_img:
        Optional 3D brain mask. If provided, computation is restricted to mask
        voxels (saves time; outside voxels are set to NaN).

    Returns
    -------
    tsnr_data:
        3D float32 array of tSNR values. NaN where std==0 or outside mask.
    tsnr_img:
        Corresponding nibabel Nifti1Image with the same affine as bold_img.
    """
    # Cast to float32 immediately to avoid double float64 memory usage
    data = bold_img.get_fdata(dtype=np.float32)  # shape (x, y, z, t)

    if mask_img is not None:
        mask = np.asarray(mask_img.dataobj, dtype=bool)
    else:
        mask = np.ones(data.shape[:3], dtype=bool)

    mean_map = np.full(data.shape[:3], np.nan, dtype=np.float32)
    std_map = np.full(data.shape[:3], np.nan, dtype=np.float32)

    mean_map[mask] = data[mask].mean(axis=1)
    std_map[mask] = data[mask].std(axis=1, ddof=1)

    # Free the large array as early as possible
    del data

    tsnr_data = np.full(mean_map.shape, np.nan, dtype=np.float32)
    valid = mask & (std_map > 0) & np.isfinite(mean_map) & np.isfinite(std_map)
    tsnr_data[valid] = mean_map[valid] / std_map[valid]

    # Clip implausible values (negative SNR or >1000 from near-zero std)
    tsnr_data = np.clip(tsnr_data, 0, 1000)
    tsnr_data[~valid] = np.nan

    tsnr_img = nib.Nifti1Image(tsnr_data, bold_img.affine, bold_img.header)
    return tsnr_data, tsnr_img


def extract_tsnr_by_roi(
    tsnr_data: np.ndarray,
    suit_data: np.ndarray,
    aseg_data: Optional[np.ndarray],
    suit_mask: Optional[np.ndarray] = None,
    global_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Extract mean tSNR per SUIT lobule and aseg cerebellar/brainstem region.

    Parameters
    ----------
    tsnr_data:
        3D float32 tSNR map (NaN for invalid voxels).
    suit_data:
        3D integer SUIT atlas array (0 = outside cerebellum).
    aseg_data:
        3D integer FreeSurfer aseg array, or None if unavailable.
    suit_mask:
        Mask applied to SUIT lobule extraction. Pass the cerebellar GM mask
        (aseg labels 8+47) to restrict to grey matter only. Falls back to
        global_mask if None.
    global_mask:
        Whole-brain binary mask (uint8 or bool). Used for aseg ROIs,
        whole-brain mean, and cerebellar mean.

    Returns
    -------
    Dict with keys:
        - 'suit_<lobule_name>': mean tSNR per SUIT lobule (GM-masked)
        - 'aseg_<region_name>': mean tSNR per aseg ROI (brain-masked)
        - 'wholebrain_mean': mean tSNR across all valid in-mask voxels
        - 'cereb_gm_mean': mean tSNR in cerebellar GM (aseg 8+47) — NaN if unavailable
        - 'cereb_mean': mean tSNR in all SUIT voxels combined (brain-masked)
        - 'cereb_wb_ratio': cereb_gm_mean / wholebrain_mean
    """
    result: Dict[str, float] = {}

    # SUIT lobule tSNR — use GM mask so only cortical voxels contribute
    suit_stats = extract_roi_stats(tsnr_data, suit_data, SUIT_LABEL_MAP, suit_mask)
    for name, val in suit_stats.items():
        result[f"suit_{name}"] = val

    # aseg ROI tSNR — use whole-brain mask
    if aseg_data is not None:
        aseg_stats = extract_roi_stats(tsnr_data, aseg_data, ASEG_LABEL_MAP, global_mask)
        for name, val in aseg_stats.items():
            result[f"aseg_{name}"] = val

    # Whole-brain mean (brain mask)
    if global_mask is not None:
        valid_mask = global_mask.astype(bool) & np.isfinite(tsnr_data)
    else:
        valid_mask = np.isfinite(tsnr_data)
    result["wholebrain_mean"] = float(np.nanmean(tsnr_data[valid_mask])) if valid_mask.any() else float("nan")

    # Cerebellar GM mean (aseg labels 8+47, GM mask)
    if suit_mask is not None:
        gm_vals = tsnr_data[suit_mask.astype(bool) & np.isfinite(tsnr_data)]
        result["cereb_gm_mean"] = float(np.nanmean(gm_vals)) if len(gm_vals) > 0 else float("nan")
    else:
        result["cereb_gm_mean"] = float("nan")

    # Cerebellar mean — all SUIT voxels, brain mask (for backward compatibility)
    cereb_mask = suit_data > 0
    if global_mask is not None:
        cereb_mask = cereb_mask & global_mask.astype(bool)
    cereb_vals = tsnr_data[cereb_mask & np.isfinite(tsnr_data)]
    result["cereb_mean"] = float(np.nanmean(cereb_vals)) if len(cereb_vals) > 0 else float("nan")

    # Ratio uses GM mean as the numerator (more meaningful than WM-diluted cereb_mean)
    wb = result["wholebrain_mean"]
    cb = result["cereb_gm_mean"] if np.isfinite(result["cereb_gm_mean"]) else result["cereb_mean"]
    result["cereb_wb_ratio"] = (cb / wb) if (np.isfinite(wb) and wb > 0 and np.isfinite(cb)) else float("nan")

    return result


def compute_lobule_coverage_quality(
    tsnr_data: np.ndarray,
    suit_data: np.ndarray,
    mask_data: Optional[np.ndarray] = None,
    min_valid_fraction: float = 0.5,
) -> Dict[str, bool]:
    """
    For each SUIT lobule, check whether >= min_valid_fraction of its in-mask voxels
    have valid (non-NaN) tSNR values.

    Returns
    -------
    Dict mapping lobule name → True if quality is acceptable.
    """
    result: Dict[str, bool] = {}
    for label_id, name in SUIT_LABEL_MAP.items():
        roi = suit_data == label_id
        if mask_data is not None:
            roi = roi & mask_data.astype(bool)
        n_total = roi.sum()
        if n_total == 0:
            result[name] = False
            continue
        n_valid = (roi & np.isfinite(tsnr_data)).sum()
        result[name] = (n_valid / n_total) >= min_valid_fraction
    return result
