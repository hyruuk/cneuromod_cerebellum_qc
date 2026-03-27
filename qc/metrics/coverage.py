"""
Brain mask coverage and signal dropout analysis per cerebellar lobule.

Checks:
- What fraction of each SUIT lobule is inside the brain mask
- What is the mean EPI signal per lobule relative to whole-brain mean
  (from boldref = 3D reference volume used in registration, reflects susceptibility dropout)
"""

from __future__ import annotations

from typing import Dict, Optional

import nibabel as nib
import numpy as np

from qc.atlas import extract_roi_coverage, extract_roi_stats, SUIT_LABEL_MAP, ASEG_LABEL_MAP


def compute_mask_coverage(
    mask_img: nib.Nifti1Image,
    suit_data: np.ndarray,
    aseg_data: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute the fraction of each SUIT lobule (and aseg region) inside the brain mask.

    Parameters
    ----------
    mask_img:
        3D brain mask NIfTI (1 = in-mask, 0 = outside).
    suit_data:
        3D SUIT atlas integer array.
    aseg_data:
        Optional 3D aseg integer array.

    Returns
    -------
    Dict with keys:
        - 'suit_coverage_<lobule>': fraction in [0, 1]
        - 'aseg_coverage_<region>': fraction in [0, 1]
        - 'suit_coverage_mean': mean coverage across all SUIT lobules
        - 'suit_coverage_min': minimum coverage (worst lobule)
        - 'suit_coverage_min_lobule': name of worst-covered lobule
    """
    mask_data = np.asarray(mask_img.dataobj, dtype=np.uint8)
    result: Dict[str, float] = {}

    # SUIT lobule coverage
    suit_cov = extract_roi_coverage(mask_data, suit_data, SUIT_LABEL_MAP)
    for name, val in suit_cov.items():
        result[f"suit_coverage_{name}"] = val

    valid_cov = {k: v for k, v in suit_cov.items() if np.isfinite(v)}
    if valid_cov:
        result["suit_coverage_mean"] = float(np.mean(list(valid_cov.values())))
        min_name = min(valid_cov, key=valid_cov.get)
        result["suit_coverage_min"] = valid_cov[min_name]
        result["suit_coverage_min_lobule"] = min_name  # stored as string; keep NaN-safe
    else:
        result["suit_coverage_mean"] = float("nan")
        result["suit_coverage_min"] = float("nan")
        result["suit_coverage_min_lobule"] = "unknown"

    # aseg cerebellar region coverage
    if aseg_data is not None:
        aseg_cov = extract_roi_coverage(mask_data, aseg_data, ASEG_LABEL_MAP)
        for name, val in aseg_cov.items():
            result[f"aseg_coverage_{name}"] = val

    return result


def compute_signal_dropout(
    boldref_img: nib.Nifti1Image,
    suit_data: np.ndarray,
    mask_data: Optional[np.ndarray],
    aseg_data: Optional[np.ndarray] = None,
    dropout_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute mean EPI signal per SUIT lobule from the boldref (3D reference volume).

    The boldref captures susceptibility-induced signal dropout patterns.
    Relative signal = lobule mean / whole-brain mean.
    Values < dropout_threshold flag the lobule as susceptibility-affected.

    Parameters
    ----------
    boldref_img:
        3D boldref NIfTI (the mean or reference EPI volume from fMRIPrep).
    suit_data:
        3D SUIT atlas integer array.
    mask_data:
        3D brain mask array (uint8 or bool).
    aseg_data:
        Optional aseg array.
    dropout_threshold:
        Lobules with relative signal < this value are flagged as dropout.

    Returns
    -------
    Dict with keys:
        - 'dropout_<lobule>': relative signal (float)
        - 'dropout_flag_<lobule>': 1.0 if flagged, 0.0 otherwise
        - 'n_dropout_lobules': count of flagged lobules
        - 'wholebrain_boldref_mean': whole-brain mean signal
    """
    boldref_data = boldref_img.get_fdata(dtype=np.float32)
    result: Dict[str, float] = {}

    if mask_data is not None:
        mask_bool = mask_data.astype(bool)
    else:
        mask_bool = np.ones(boldref_data.shape, dtype=bool)

    # Whole-brain mean (within mask, robust)
    wb_vals = boldref_data[mask_bool]
    wb_vals = wb_vals[np.isfinite(wb_vals) & (wb_vals > 0)]
    wb_mean = float(np.median(wb_vals)) if len(wb_vals) > 0 else float("nan")
    result["wholebrain_boldref_mean"] = wb_mean

    # Per-lobule relative signal
    n_dropout = 0
    suit_stats = extract_roi_stats(boldref_data, suit_data, SUIT_LABEL_MAP, mask_bool)
    for name, mean_val in suit_stats.items():
        if np.isfinite(wb_mean) and wb_mean > 0 and np.isfinite(mean_val):
            rel = mean_val / wb_mean
        else:
            rel = float("nan")
        result[f"dropout_{name}"] = rel
        flagged = float(np.isfinite(rel) and rel < dropout_threshold)
        result[f"dropout_flag_{name}"] = flagged
        if flagged:
            n_dropout += 1

    result["n_dropout_lobules"] = float(n_dropout)

    # aseg regions if available
    if aseg_data is not None:
        aseg_stats = extract_roi_stats(boldref_data, aseg_data, ASEG_LABEL_MAP, mask_bool)
        for name, mean_val in aseg_stats.items():
            if np.isfinite(wb_mean) and wb_mean > 0 and np.isfinite(mean_val):
                rel = mean_val / wb_mean
            else:
                rel = float("nan")
            result[f"aseg_dropout_{name}"] = rel

    return result
