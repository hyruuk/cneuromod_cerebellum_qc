"""
Synthetic test fixtures for the cerebellar QC pipeline.

Creates minimal NIfTI images and DataFrames that mimic the real dataset
structure, allowing tests to run without downloading any data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import nibabel as nib

# Standard affine for synthetic 2mm MNI-like data
AFFINE_2MM = np.array([
    [-2., 0., 0., 90.],
    [0., 2., 0., -126.],
    [0., 0., 2., -72.],
    [0., 0., 0., 1.],
])

SHAPE_3D = (20, 20, 20)
N_TRS = 50


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_bold(rng) -> nib.Nifti1Image:
    """4D int32 BOLD with signal ~1000 ± Gaussian noise (realistic SNR ~20)."""
    data = (1000 + rng.normal(0, 50, (*SHAPE_3D, N_TRS))).astype(np.int32)
    return nib.Nifti1Image(data, AFFINE_2MM)


@pytest.fixture
def synthetic_bold_lowsnr(rng) -> nib.Nifti1Image:
    """4D int32 BOLD with much higher noise (simulating susceptibility dropout)."""
    data = (500 + rng.normal(0, 250, (*SHAPE_3D, N_TRS))).astype(np.int32)
    return nib.Nifti1Image(data, AFFINE_2MM)


@pytest.fixture
def synthetic_mask() -> nib.Nifti1Image:
    """Full brain mask (all 1s)."""
    data = np.ones(SHAPE_3D, dtype=np.uint8)
    return nib.Nifti1Image(data, AFFINE_2MM)


@pytest.fixture
def synthetic_partial_mask() -> nib.Nifti1Image:
    """Partial brain mask — only central 10x10x10 voxels (simulates truncation)."""
    data = np.zeros(SHAPE_3D, dtype=np.uint8)
    data[5:15, 5:15, 5:15] = 1
    return nib.Nifti1Image(data, AFFINE_2MM)


@pytest.fixture
def synthetic_suit_atlas() -> np.ndarray:
    """
    Synthetic SUIT atlas: 4 lobules assigned to different quadrants.
    Labels 1-4 cover 5x10x10 voxels each; rest is 0 (outside cerebellum).
    """
    data = np.zeros(SHAPE_3D, dtype=np.int16)
    data[0:5, :, :] = 1    # Left_I_IV
    data[5:10, :, :] = 2   # Right_I_IV
    data[10:15, :, :] = 3  # Left_V
    data[15:20, :, :] = 4  # Right_V
    return data


@pytest.fixture
def synthetic_suit_atlas_full() -> np.ndarray:
    """
    Synthetic SUIT atlas with all 34 regions, each with ~1 voxel.
    Used for testing label iteration without needing real coverage.
    """
    data = np.zeros(SHAPE_3D, dtype=np.int16)
    rng = np.random.default_rng(99)
    coords = rng.integers(0, 20, size=(34, 3))
    for label_id, (x, y, z) in enumerate(coords, start=1):
        data[x, y, z] = label_id
    return data


@pytest.fixture
def synthetic_aseg() -> np.ndarray:
    """
    Synthetic FreeSurfer aseg: assigns known label IDs to regions.
    Labels: 8 (L-Cereb-Cortex), 47 (R-Cereb-Cortex), 15 (4th-Ventricle),
            16 (Brain-Stem), 7 (L-Cereb-WM), 46 (R-Cereb-WM).
    """
    data = np.zeros(SHAPE_3D, dtype=np.int32)
    data[0:5, 0:10, :] = 8    # Left cerebellar cortex
    data[5:10, 0:10, :] = 47  # Right cerebellar cortex
    data[10:13, 10:13, 10:13] = 15  # 4th ventricle
    data[13:16, :, :] = 16   # Brain stem
    data[0:5, 10:20, :] = 7  # Left cerebellar WM
    data[5:10, 10:20, :] = 46  # Right cerebellar WM
    return data


@pytest.fixture
def synthetic_boldref(rng) -> nib.Nifti1Image:
    """3D boldref with some regional dropout (low signal in one quadrant)."""
    data = np.full(SHAPE_3D, 1000.0, dtype=np.float32)
    # Simulate dropout in first quadrant (x < 5)
    data[:5, :, :] = 200.0
    data += rng.normal(0, 10, SHAPE_3D)
    return nib.Nifti1Image(data, AFFINE_2MM)


@pytest.fixture
def synthetic_confounds(rng) -> pd.DataFrame:
    """
    Realistic confounds DataFrame with FD, DVARS, motion params, and aCompCor.
    First FD value is NaN (as in fMRIPrep output).
    """
    fd = np.concatenate([[np.nan], np.abs(rng.normal(0.15, 0.12, N_TRS - 1))])
    # Add a few high-motion spikes
    fd[10] = 0.8
    fd[30] = 1.2

    dvars = np.abs(rng.normal(1.0, 0.3, N_TRS))
    std_dvars = np.abs(rng.normal(1.0, 0.25, N_TRS))
    std_dvars[10] = 2.0  # spike aligned with high FD

    df = pd.DataFrame(
        {
            "framewise_displacement": fd,
            "dvars": dvars,
            "std_dvars": std_dvars,
            "rmsd": np.abs(rng.normal(0.12, 0.08, N_TRS)),
            "trans_x": rng.normal(0, 0.3, N_TRS),
            "trans_y": rng.normal(0, 0.3, N_TRS),
            "trans_z": rng.normal(0, 0.3, N_TRS),
            "rot_x": rng.normal(0, 0.005, N_TRS),
            "rot_y": rng.normal(0, 0.005, N_TRS),
            "rot_z": rng.normal(0, 0.005, N_TRS),
            "global_signal": rng.normal(1000, 50, N_TRS),
            "csf": rng.normal(800, 40, N_TRS),
            "white_matter": rng.normal(900, 30, N_TRS),
            **{f"a_comp_cor_{i:02d}": rng.normal(0, 1, N_TRS) for i in range(5)},
        }
    )
    return df


@pytest.fixture
def synthetic_confounds_json() -> dict:
    """
    Minimal confounds JSON metadata simulating fMRIPrep 20.2.5 output.
    Includes 5 retained aCompCor components per mask type.
    """
    meta = {}
    for mask_type in ("CSF", "WM", "combined"):
        cumvar = 0.0
        for i in range(5):
            var = 0.1 - i * 0.01  # decreasing variance per component
            cumvar += var
            meta[f"a_comp_cor_{i:02d}_{mask_type}"] = {
                "CumulativeVarianceExplained": round(cumvar, 6),
                "Mask": mask_type,
                "Method": "aCompCor",
                "Retained": True,
                "SingularValue": 100.0 - i * 10,
                "VarianceExplained": round(var, 6),
            }
        # Add some dropped components
        for i in range(5, 8):
            meta[f"dropped_{i}_{mask_type}"] = {
                "CumulativeVarianceExplained": round(cumvar + 0.001 * (i - 4), 6),
                "Mask": mask_type,
                "Method": "aCompCor",
                "Retained": False,
                "SingularValue": 50.0 - i,
                "VarianceExplained": 0.001,
            }

    # Add non-aCompCor entries
    meta["csf"] = {"Method": "Mean"}
    meta["global_signal"] = {"Method": "Mean"}
    return meta
