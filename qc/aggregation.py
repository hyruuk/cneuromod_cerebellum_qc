"""
Per-run processing pipeline and result aggregation.

process_run():  Full pipeline for a single run → result dict
aggregate_results():  List of run dicts → DataFrames for the report
"""

from __future__ import annotations

import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from qc.discovery import RunInfo
from qc.atlas import (
    SUIT_LABEL_MAP,
    ASEG_LABEL_MAP,
    YEO7_LABEL_MAP,
    load_subject_aseg,
    extract_roi_stats,
)
from qc.metrics.tsnr import compute_tsnr_map, extract_tsnr_by_roi
from qc.metrics.motion import (
    load_confounds,
    compute_motion_metrics,
    parse_acompcor_stats,
    get_motion_params,
)
from qc.metrics.coverage import compute_mask_coverage, compute_signal_dropout
from qc.metrics.noise import (
    compute_csf_correlation,
    compute_ar1,
    compute_motor_cereb_correlation,
    compute_signal_drift,
    extract_carpet_data,
    compute_interlobule_correlation,
)


def process_run(
    run_info: RunInfo,
    suit_data: np.ndarray,
    aseg_data: Optional[np.ndarray],
    aseg_img: Optional[nib.Nifti1Image],
    yeo_data: Optional[np.ndarray] = None,
    fd_threshold: float = 0.5,
    dvars_threshold: float = 1.5,
    no_bold: bool = False,
    extract_carpet: bool = False,
    extract_interlobule: bool = False,
) -> dict:
    """
    Run the full QC pipeline for a single functional run.

    Parameters
    ----------
    run_info:
        RunInfo dataclass for this run.
    suit_data:
        Pre-loaded SUIT atlas array (shared across all runs).
    aseg_data:
        Pre-loaded aseg array for this subject, or None.
    aseg_img:
        nibabel aseg image (for resampling reference), or None.
    yeo_data:
        Pre-loaded Yeo 7-network atlas array (resampled to reference grid), or None.
    fd_threshold:
        FD scrubbing threshold (mm).
    dvars_threshold:
        std_dvars scrubbing threshold.
    no_bold:
        If True, skip BOLD loading (confounds-only mode).
    extract_carpet:
        If True, extract and return carpet plot data.
    extract_interlobule:
        If True, compute inter-lobule correlation matrix.

    Returns
    -------
    Flat dict of QC metrics + metadata. Missing values are np.nan.
    """
    result = {
        "subject": run_info.subject,
        "session": run_info.session,
        "run": run_info.run,
        "tr": run_info.tr,
        "bold_available": float(run_info.bold_available),
        "confounds_available": float(run_info.confounds_available),
        "aseg_available": float(run_info.aseg_available),
        "mask_available": float(run_info.mask_available),
        "boldref_available": float(run_info.boldref_available),
    }

    # --- Confounds (motion, aCompCor) ---
    if run_info.confounds_available:
        try:
            df, meta = load_confounds(run_info.confounds_path, run_info.confounds_json_path)
            motion_metrics = compute_motion_metrics(df, fd_threshold, dvars_threshold)
            result.update(motion_metrics)
            acompcor_metrics = parse_acompcor_stats(meta)
            result.update(acompcor_metrics)
            # Store FD series for carpet plot overlay (kept as key for report)
            result["_fd_series"] = df["framewise_displacement"].values.astype(float) if "framewise_displacement" in df.columns else None
            motion_params = get_motion_params(df)
        except Exception as e:
            warnings.warn(f"[{run_info.subject}/{run_info.session}/{run_info.run}] Confounds error: {e}")
            motion_params = None
    else:
        motion_params = None

    if no_bold:
        result["_carpet"] = None
        result["_interlobule_corr"] = None
        result["_interlobule_names"] = None
        result["_gm_display_mask"] = None
        return result

    # --- BOLD-based metrics ---
    if not run_info.bold_available:
        result["_carpet"] = None
        result["_interlobule_corr"] = None
        result["_interlobule_names"] = None
        result["_gm_display_mask"] = None
        return result

    try:
        bold_img = nib.load(run_info.bold_path)
        # Cast to float32 immediately to minimize memory
        bold_data = bold_img.get_fdata(dtype=np.float32)
    except Exception as e:
        warnings.warn(f"[{run_info.subject}/{run_info.session}/{run_info.run}] BOLD load error: {e}")
        result["_carpet"] = None
        result["_interlobule_corr"] = None
        result["_interlobule_names"] = None
        result["_gm_display_mask"] = None
        return result

    # Load mask
    mask_data = None
    mask_img = None
    if run_info.mask_available:
        try:
            mask_img = nib.load(run_info.mask_path)
            mask_data = np.asarray(mask_img.dataobj, dtype=np.uint8)
        except Exception as e:
            warnings.warn(f"[{run_info.subject}/{run_info.session}/{run_info.run}] Mask load error: {e}")

    # Resample suit_data to BOLD grid if shapes differ
    suit_resampled = _resample_atlas_to_bold(suit_data, bold_img)
    aseg_resampled = None
    if aseg_data is not None and aseg_img is not None:
        aseg_resampled = _resample_atlas_to_bold(aseg_data, bold_img, aseg_img)

    # Cerebellar grey matter mask: aseg labels 8 (L-Cereb-Cortex) + 47 (R-Cereb-Cortex),
    # intersected with the brain mask. Used for SUIT lobule tSNR to exclude WM voxels.
    gm_cereb_mask = None
    if aseg_resampled is not None:
        gm_cereb_mask = np.isin(aseg_resampled, [8, 47]).astype(np.uint8)
        if mask_data is not None:
            gm_cereb_mask = (gm_cereb_mask & mask_data.astype(bool)).astype(np.uint8)

    # Cerebral cortex grey matter mask: aseg labels 3 (L-Cerebral-Cortex) + 42 (R-Cerebral-Cortex).
    # Used for Yeo network tSNR to restrict to cortical GM only.
    gm_cortex_mask = None
    if aseg_resampled is not None:
        gm_cortex_mask = np.isin(aseg_resampled, [3, 42]).astype(np.uint8)
        if mask_data is not None:
            gm_cortex_mask = (gm_cortex_mask & mask_data.astype(bool)).astype(np.uint8)

    # Resample Yeo atlas to BOLD grid if needed
    yeo_resampled = None
    if yeo_data is not None:
        yeo_resampled = _resample_atlas_to_bold(yeo_data, bold_img)

    # --- tSNR ---
    try:
        tsnr_data, tsnr_img = compute_tsnr_map(bold_img, mask_img)
        # SUIT lobule metrics use GM mask (cerebellar cortex only);
        # aseg and whole-brain metrics keep the full brain mask.
        tsnr_metrics = extract_tsnr_by_roi(
            tsnr_data, suit_resampled, aseg_resampled,
            suit_mask=gm_cereb_mask if gm_cereb_mask is not None else mask_data,
            global_mask=mask_data,
        )
        result.update(tsnr_metrics)
        result["_tsnr_img"] = tsnr_img

        # Combined GM mask for display: cerebral cortex (3+42) + cerebellar cortex (8+47)
        # Used to mask the tSNR viewer so only GM is shown.
        if gm_cereb_mask is not None or gm_cortex_mask is not None:
            gm_display = np.zeros(tsnr_data.shape, dtype=bool)
            if gm_cereb_mask is not None:
                gm_display |= gm_cereb_mask.astype(bool)
            if gm_cortex_mask is not None:
                gm_display |= gm_cortex_mask.astype(bool)
            result["_gm_display_mask"] = gm_display
        else:
            result["_gm_display_mask"] = None

        # Yeo network tSNR — use cortical GM mask when available, fall back to brain mask
        if yeo_resampled is not None:
            yeo_mask = (
                gm_cortex_mask
                if (gm_cortex_mask is not None and gm_cortex_mask.any())
                else mask_data
            )
            yeo_stats = extract_roi_stats(tsnr_data, yeo_resampled, YEO7_LABEL_MAP, yeo_mask)
            for name, val in yeo_stats.items():
                result[f"yeo_{name}"] = val
    except Exception as e:
        warnings.warn(f"[{run_info.subject}/{run_info.session}/{run_info.run}] tSNR error: {e}")
        result["_tsnr_img"] = None
        result["_gm_display_mask"] = None

    # --- Coverage (mask coverage + dropout) ---
    if mask_img is not None:
        try:
            cov_metrics = compute_mask_coverage(mask_img, suit_resampled, aseg_resampled)
            result.update(cov_metrics)
        except Exception as e:
            warnings.warn(f"[{run_info.subject}/{run_info.session}/{run_info.run}] Coverage error: {e}")

    if run_info.boldref_available:
        try:
            boldref_img = nib.load(run_info.boldref_path)
            dropout_metrics = compute_signal_dropout(boldref_img, suit_resampled, mask_data, aseg_resampled)
            result.update(dropout_metrics)
        except Exception as e:
            warnings.warn(f"[{run_info.subject}/{run_info.session}/{run_info.run}] Dropout error: {e}")

    # --- Noise: CSF correlation ---
    if aseg_resampled is not None:
        try:
            csf_metrics = compute_csf_correlation(bold_data, aseg_resampled, mask_data)
            result.update(csf_metrics)
        except Exception as e:
            warnings.warn(f"[{run_info.subject}/{run_info.session}/{run_info.run}] CSF corr error: {e}")

    # --- Noise: AR(1) ---
    if aseg_resampled is not None:
        try:
            ar1_metrics = compute_ar1(bold_data, aseg_resampled, mask_data, motion_params)
            result.update(ar1_metrics)
        except Exception as e:
            warnings.warn(f"[{run_info.subject}/{run_info.session}/{run_info.run}] AR1 error: {e}")

    # --- Noise: Motor cortex–cerebellum correlation ---
    if aseg_resampled is not None:
        try:
            motor_metrics = compute_motor_cereb_correlation(
                bold_data, aseg_resampled, mask_data, bold_affine=bold_img.affine
            )
            result.update(motor_metrics)
        except Exception as e:
            warnings.warn(f"[{run_info.subject}/{run_info.session}/{run_info.run}] Motor corr error: {e}")

    # --- Signal drift ---
    if aseg_resampled is not None:
        try:
            drift_metrics = compute_signal_drift(bold_data, aseg_resampled, mask_data, run_info.tr)
            result.update(drift_metrics)
        except Exception as e:
            warnings.warn(f"[{run_info.subject}/{run_info.session}/{run_info.run}] Drift error: {e}")

    # --- Carpet plot (optional, only for representative runs) ---
    if extract_carpet:
        try:
            carpet = extract_carpet_data(bold_data, suit_resampled, mask_data)
            result["_carpet"] = carpet
        except Exception as e:
            warnings.warn(f"[{run_info.subject}/{run_info.session}/{run_info.run}] Carpet error: {e}")
            result["_carpet"] = None
    else:
        result["_carpet"] = None

    # --- Inter-lobule correlation matrix (optional) ---
    if extract_interlobule:
        try:
            corr_matrix, lobule_names = compute_interlobule_correlation(bold_data, suit_resampled, mask_data)
            result["_interlobule_corr"] = corr_matrix
            result["_interlobule_names"] = lobule_names
        except Exception as e:
            warnings.warn(f"[{run_info.subject}/{run_info.session}/{run_info.run}] Interlobule corr error: {e}")
            result["_interlobule_corr"] = None
            result["_interlobule_names"] = None
    else:
        result["_interlobule_corr"] = None
        result["_interlobule_names"] = None

    # Explicit memory cleanup
    del bold_data

    return result


def _resample_atlas_to_bold(
    atlas_data: np.ndarray,
    bold_img: nib.Nifti1Image,
    atlas_img: Optional[nib.Nifti1Image] = None,
) -> np.ndarray:
    """
    Return atlas_data if shapes match, otherwise resample to bold_img grid.

    This is a lightweight check — proper resampling is done once in atlas.py.
    If shapes differ (shouldn't happen in normal use), use nearest-neighbour
    interpolation via nilearn.
    """
    bold_shape = bold_img.shape[:3]
    if atlas_data.shape == bold_shape:
        return atlas_data

    from nilearn.image import resample_to_img as _resample
    import nibabel as nib

    # Wrap in a NIfTI to resample
    if atlas_img is not None:
        atlas_nib = nib.Nifti1Image(atlas_data, atlas_img.affine)
    else:
        # Assume same affine as bold (should never happen in practice)
        atlas_nib = nib.Nifti1Image(atlas_data, bold_img.affine)

    bold_ref = nib.Nifti1Image(np.zeros(bold_shape), bold_img.affine)
    resampled = _resample(atlas_nib, bold_ref, interpolation="nearest")
    result = np.asarray(resampled.dataobj, dtype=atlas_data.dtype)
    # Squeeze spurious trailing dim (NIfTI dim[4]=1 artefact)
    if result.ndim == 4 and result.shape[3] == 1:
        result = result[..., 0]
    return result


def identify_representative_runs(
    run_results: List[dict],
) -> Dict[str, Dict[str, str]]:
    """
    For each subject, identify the worst-motion and best-motion run.

    Returns
    -------
    Dict: {subject: {'worst': 'ses-XXX/run-N', 'best': 'ses-XXX/run-N'}}
    """
    by_subject: Dict[str, list] = {}
    for r in run_results:
        subj = r["subject"]
        if subj not in by_subject:
            by_subject[subj] = []
        mean_fd = r.get("mean_fd", float("nan"))
        if np.isfinite(mean_fd):
            by_subject[subj].append((mean_fd, r["session"], r["run"]))

    result: Dict[str, Dict[str, str]] = {}
    for subj, items in by_subject.items():
        if not items:
            continue
        items_sorted = sorted(items)
        best_fd, best_ses, best_run = items_sorted[0]
        worst_fd, worst_ses, worst_run = items_sorted[-1]
        result[subj] = {
            "best": f"{best_ses}/{best_run}",
            "worst": f"{worst_ses}/{worst_run}",
        }
    return result


def aggregate_results(run_results: List[dict]) -> Dict[str, pd.DataFrame]:
    """
    Convert list of per-run result dicts into aggregated DataFrames.

    Private keys (starting with '_') are excluded from the DataFrames.

    Returns
    -------
    Dict with keys:
        'runs': one row per run
        'sessions': one row per (subject, session) — averaged over runs
        'subjects': one row per subject — averaged over all runs
    """
    # Strip private keys
    public_results = []
    for r in run_results:
        public = {k: v for k, v in r.items() if not k.startswith("_")}
        # session_num for sorting
        import re
        m = re.search(r"ses-(\d+)", public.get("session", ""))
        public["session_num"] = int(m.group(1)) if m else 0
        m2 = re.search(r"run-(\d+)", public.get("run", ""))
        public["run_num"] = int(m2.group(1)) if m2 else 0
        public_results.append(public)

    df_runs = pd.DataFrame(public_results)
    if df_runs.empty:
        return {"runs": df_runs, "sessions": pd.DataFrame(), "subjects": pd.DataFrame()}

    # Sort
    df_runs = df_runs.sort_values(["subject", "session_num", "run_num"]).reset_index(drop=True)

    # Numeric columns only for aggregation
    numeric_cols = df_runs.select_dtypes(include=[np.number]).columns.tolist()
    exclude_from_agg = ["session_num", "run_num", "bold_available", "confounds_available",
                        "aseg_available", "mask_available", "boldref_available", "n_trs"]

    agg_cols = [c for c in numeric_cols if c not in exclude_from_agg]

    # Session-level aggregation (mean over runs within session)
    df_sessions = (
        df_runs.groupby(["subject", "session", "session_num"])[agg_cols]
        .mean()
        .reset_index()
        .sort_values(["subject", "session_num"])
        .reset_index(drop=True)
    )

    # Subject-level aggregation (mean over all runs)
    df_subjects = (
        df_runs.groupby("subject")[agg_cols]
        .mean()
        .reset_index()
    )

    return {
        "runs": df_runs,
        "sessions": df_sessions,
        "subjects": df_subjects,
    }
