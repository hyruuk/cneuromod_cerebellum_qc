"""
Atlas loading and ROI extraction utilities.

Handles:
- SUIT cerebellar atlas — Diedrichsen 2009 Anatom parcellation (34 regions, MNI space)
- FreeSurfer aseg segmentation (per-subject, loaded once)
- Resampling to BOLD grid
- Label maps for both atlases
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------

# Fallback label map — matches atl-Anatom.tsv from DiedrichsenLab/cerebellar_atlases.
# load_suit_atlas() will override this from the downloaded TSV when available.
SUIT_LABEL_MAP: Dict[int, str] = {
    1: "Left_I_IV",
    2: "Right_I_IV",
    3: "Left_V",
    4: "Right_V",
    5: "Left_VI",
    6: "Vermis_VI",
    7: "Right_VI",
    8: "Left_CrusI",
    9: "Vermis_CrusI",
    10: "Right_CrusI",
    11: "Left_CrusII",
    12: "Vermis_CrusII",
    13: "Right_CrusII",
    14: "Left_VIIb",
    15: "Vermis_VIIb",
    16: "Right_VIIb",
    17: "Left_VIIIa",
    18: "Vermis_VIIIa",
    19: "Right_VIIIa",
    20: "Left_VIIIb",
    21: "Vermis_VIIIb",
    22: "Right_VIIIb",
    23: "Left_IX",
    24: "Vermis_IX",
    25: "Right_IX",
    26: "Left_X",
    27: "Vermis_X",
    28: "Right_X",
    29: "Left_Dentate",
    30: "Right_Dentate",
    31: "Left_Interposed",
    32: "Right_Interposed",
    33: "Left_Fastigial",
    34: "Right_Fastigial",
}

# FreeSurfer aseg labels relevant for cerebellum / brainstem / ventricles
ASEG_LABEL_MAP: Dict[int, str] = {
    7: "L-Cereb-WM",
    8: "L-Cereb-Cortex",
    15: "4th-Ventricle",
    16: "Brain-Stem",
    46: "R-Cereb-WM",
    47: "R-Cereb-Cortex",
}

# Labels used for the CSF contamination check
ASEG_CSF_LABELS = {15: "4th-Ventricle", 4: "L-Lat-Ventricle", 43: "R-Lat-Ventricle"}

# Motor cortex: precentral gyrus labels in aparc+aseg (Desikan-Killiany)
# These are Freesurfer 2009 labels for precentral gyrus
ASEG_MOTOR_LABELS = {1024: "L-precentral", 2024: "R-precentral"}

# Combined cerebellar cortex labels (for inter-region correlation)
ASEG_CEREB_CORTEX_LABELS = {8: "L-Cereb-Cortex", 47: "R-Cereb-Cortex"}

# Cerebral cortex labels (for cortical GM masking of Yeo network tSNR)
ASEG_CORTEX_LABELS = {3: "L-Cerebral-Cortex", 42: "R-Cerebral-Cortex"}

# SUIT deep nuclei — too small at BOLD resolution for reliable timeseries
SUIT_DEEP_NUCLEI = {"Left_Dentate", "Right_Dentate", "Left_Interposed", "Right_Interposed",
                    "Left_Fastigial", "Right_Fastigial"}

# Yeo 7-network parcellation label map
YEO7_LABEL_MAP: Dict[int, str] = {
    1: "Visual",
    2: "Somatomotor",
    3: "DorsAttn",
    4: "SalVentAttn",
    5: "Limbic",
    6: "Control",
    7: "Default",
}


# ---------------------------------------------------------------------------
# SUIT atlas loading
# ---------------------------------------------------------------------------

def _tsv_path_for_atlas(suit_path: Path) -> Path:
    """Derive the expected TSV label path: strip .nii/.nii.gz, append .tsv."""
    stem = suit_path.name
    for suffix in (".gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return suit_path.parent / (stem + ".tsv")


def _load_label_map_from_tsv(tsv_path: Path) -> Dict[int, str]:
    """Parse a BIDS atlas TSV (columns: index, name, ...) into {int: str}."""
    import csv
    label_map: Dict[int, str] = {}
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            idx = int(row["index"])
            if idx == 0:
                continue  # skip background
            label_map[idx] = row["name"]
    return label_map


def load_suit_atlas(
    suit_path: str | Path,
    reference_img: nib.Nifti1Image,
    cache_dir: Optional[str | Path] = None,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Load and resample the SUIT cerebellar atlas to the BOLD grid.

    Parameters
    ----------
    suit_path:
        Path to the SUIT lobular atlas NIfTI file (discrete integer labels).
    reference_img:
        A nibabel image in the target space (e.g., a BOLD volume or brain mask).
        Used to define the resampling target grid.
    cache_dir:
        If provided, the resampled atlas is saved here to avoid re-resampling.

    Returns
    -------
    suit_data:
        3D integer array of SUIT label IDs, resampled to BOLD grid.
    label_map:
        Dict mapping integer IDs → lobule name strings.
    """
    suit_path = Path(suit_path)
    cache_dir = Path(cache_dir) if cache_dir else suit_path.parent

    # Load label map from TSV alongside the atlas; fall back to hardcoded map.
    # Updating SUIT_LABEL_MAP in-place propagates to all modules that imported it.
    tsv_path = _tsv_path_for_atlas(suit_path)
    if tsv_path.exists():
        label_map = _load_label_map_from_tsv(tsv_path)
        SUIT_LABEL_MAP.clear()
        SUIT_LABEL_MAP.update(label_map)
    else:
        label_map = SUIT_LABEL_MAP
        warnings.warn(
            f"No label TSV found at {tsv_path}. Using hardcoded fallback label map. "
            "Run download_suit_atlas.py to obtain the authoritative labels.",
            UserWarning,
            stacklevel=2,
        )

    # Check for cached resampled version
    cached = cache_dir / f"suit_resampled_{reference_img.shape[0]}x{reference_img.shape[1]}x{reference_img.shape[2]}.nii.gz"
    if cached.exists():
        suit_img = nib.load(cached)
    else:
        suit_img_orig = nib.load(suit_path)
        suit_img = resample_to_img(
            suit_img_orig,
            reference_img,
            interpolation="nearest",
            copy_header=True,
        )
        nib.save(suit_img, cached)

    suit_data = np.asarray(suit_img.dataobj, dtype=np.int16)

    # Validate: check which label IDs are present and warn about missing ones
    present_labels = set(np.unique(suit_data)) - {0}
    expected_labels = set(label_map.keys())
    missing = expected_labels - present_labels
    if missing:
        warnings.warn(
            f"SUIT atlas missing {len(missing)} expected labels after resampling: "
            f"{sorted(missing)}. This may indicate a space mismatch.",
            UserWarning,
            stacklevel=2,
        )

    # Warn about lobules with very few voxels
    for label_id, name in label_map.items():
        n_vox = (suit_data == label_id).sum()
        if 0 < n_vox < 5:
            warnings.warn(
                f"SUIT lobule {name} (id={label_id}) has only {n_vox} voxels "
                "after resampling. Metrics for this region will be unreliable.",
                UserWarning,
                stacklevel=2,
            )

    return suit_data, label_map


def load_subject_aseg(
    func_dir: Path,
    subject: str,
    session: str,
    space: str = "MNI152NLin2009cAsym",
) -> Tuple[Optional[np.ndarray], Optional[nib.Nifti1Image]]:
    """
    Load the FreeSurfer aseg segmentation for a subject.

    The aseg file is per-subject (all runs share the same git-annex object),
    so we only need one matching file from any run.

    Returns
    -------
    aseg_data:
        3D integer array of aseg label IDs, or None if unavailable.
    aseg_img:
        nibabel image (for use as reference grid), or None if unavailable.
    """
    # Find any available aseg file for this subject/session (task/run-agnostic)
    for candidate in sorted(func_dir.glob(
        f"{subject}_{session}_task-*_run-*_space-{space}_desc-aseg_dseg.nii.gz"
    )):
        if _is_available(candidate):
            img = nib.load(candidate)
            data = np.asarray(img.dataobj, dtype=np.int32)
            return data, img

    return None, None


def _is_available(path: Path) -> bool:
    try:
        real = path.resolve()
        return real.exists() and real.stat().st_size > 0
    except (OSError, PermissionError):
        return False


# ---------------------------------------------------------------------------
# Yeo 7-network atlas loading
# ---------------------------------------------------------------------------

def load_yeo_atlas(
    reference_img: nib.Nifti1Image,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Load the Yeo 2011 7-network cortical atlas and resample to the BOLD grid.

    Uses nilearn's built-in fetcher (downloads once to ~/.nilearn/data/).
    The atlas is in MNI152 space (1 mm); resampled to BOLD space with
    nearest-neighbour interpolation.

    Parameters
    ----------
    reference_img:
        A nibabel image in the target space (e.g., a brain mask or BOLD volume).

    Returns
    -------
    yeo_data:
        3D int16 array of Yeo network IDs (0 = background), resampled to BOLD grid.
    label_map:
        Dict mapping integer IDs → network name strings.
    """
    from nilearn.datasets import fetch_atlas_yeo_2011

    yeo = fetch_atlas_yeo_2011()
    yeo_img = nib.load(yeo.thick_7)
    resampled = resample_to_img(yeo_img, reference_img, interpolation="nearest", copy_header=True)
    yeo_data = np.asarray(resampled.dataobj, dtype=np.int16)
    return yeo_data, YEO7_LABEL_MAP


# ---------------------------------------------------------------------------
# ROI statistics
# ---------------------------------------------------------------------------

def extract_roi_stats(
    data_3d: np.ndarray,
    atlas_data: np.ndarray,
    label_map: Dict[int, str],
    mask_data: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute the mean of data_3d within each ROI defined by atlas_data.

    Parameters
    ----------
    data_3d:
        3D float array (e.g., tSNR map). May contain NaN for invalid voxels.
    atlas_data:
        3D integer array of ROI labels.
    label_map:
        Dict mapping integer label IDs → region name strings.
    mask_data:
        Optional 3D boolean/uint8 array; if provided, restrict to in-mask voxels.

    Returns
    -------
    Dict mapping region name → mean value (NaN if region is empty or all-NaN).
    """
    result: Dict[str, float] = {}
    for label_id, name in label_map.items():
        roi_mask = atlas_data == label_id
        if mask_data is not None:
            roi_mask = roi_mask & mask_data.astype(bool)
        if not roi_mask.any():
            result[name] = float("nan")
            continue
        vals = data_3d[roi_mask]
        # Remove NaN (e.g., from std==0 in tSNR computation)
        vals = vals[np.isfinite(vals)]
        result[name] = float(np.nanmean(vals)) if len(vals) > 0 else float("nan")
    return result


def extract_roi_coverage(
    mask_data: np.ndarray,
    atlas_data: np.ndarray,
    label_map: Dict[int, str],
) -> Dict[str, float]:
    """
    Compute fraction of atlas voxels that are inside the brain mask per ROI.

    Returns
    -------
    Dict mapping region name → fraction in [0, 1].
    """
    result: Dict[str, float] = {}
    mask_bool = mask_data.astype(bool)
    for label_id, name in label_map.items():
        atlas_mask = atlas_data == label_id
        n_total = atlas_mask.sum()
        if n_total == 0:
            result[name] = float("nan")
            continue
        n_in_mask = (atlas_mask & mask_bool).sum()
        result[name] = float(n_in_mask) / float(n_total)
    return result


def compute_lobule_timeseries(
    bold_data: np.ndarray,
    suit_data: np.ndarray,
    label_map: Dict[int, str],
    mask_data: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract mean time series per SUIT lobule from 4D BOLD data.

    Parameters
    ----------
    bold_data:
        4D float32 array (x, y, z, t).
    suit_data:
        3D integer atlas array.
    label_map:
        SUIT label map.
    mask_data:
        Optional brain mask.

    Returns
    -------
    Dict mapping lobule name → 1D time series array.
    """
    result: Dict[str, np.ndarray] = {}
    for label_id, name in label_map.items():
        roi_mask = suit_data == label_id
        if mask_data is not None:
            roi_mask = roi_mask & mask_data.astype(bool)
        if not roi_mask.any():
            result[name] = np.full(bold_data.shape[3], np.nan)
            continue
        # Mean over spatial dims, shape (n_trs,)
        ts = bold_data[roi_mask, :].mean(axis=0)
        result[name] = ts.astype(np.float32)
    return result
