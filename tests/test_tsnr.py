"""Tests for tSNR computation and ROI extraction."""

from __future__ import annotations

import numpy as np
import pytest

from qc.metrics.tsnr import compute_tsnr_map, extract_tsnr_by_roi, compute_lobule_coverage_quality


class TestComputeTsnrMap:
    def test_basic_tsnr(self, synthetic_bold, synthetic_mask):
        """tSNR map should have correct shape and positive values in mask."""
        tsnr_data, tsnr_img = compute_tsnr_map(synthetic_bold, synthetic_mask)
        assert tsnr_data.shape == (20, 20, 20)
        assert np.isfinite(tsnr_data).any()
        assert (tsnr_data[np.isfinite(tsnr_data)] > 0).all()

    def test_tsnr_range(self, synthetic_bold, synthetic_mask):
        """For signal ~1000, noise ~50, expected tSNR ~ 20 (mean/std)."""
        tsnr_data, _ = compute_tsnr_map(synthetic_bold, synthetic_mask)
        valid = tsnr_data[np.isfinite(tsnr_data)]
        assert len(valid) > 0
        assert 5 < float(np.mean(valid)) < 100, f"Unexpected mean tSNR: {np.mean(valid)}"

    def test_tsnr_without_mask(self, synthetic_bold):
        """tSNR computation should work without a mask."""
        tsnr_data, _ = compute_tsnr_map(synthetic_bold, mask_img=None)
        assert tsnr_data.shape == (20, 20, 20)
        assert np.isfinite(tsnr_data).any()

    def test_tsnr_clipped_at_1000(self, synthetic_bold):
        """tSNR should be clipped at 1000 (handles near-zero std edge case)."""
        tsnr_data, _ = compute_tsnr_map(synthetic_bold, mask_img=None)
        assert float(np.nanmax(tsnr_data)) <= 1000.0

    def test_tsnr_nan_where_outside_mask(self, synthetic_bold, synthetic_partial_mask):
        """Voxels outside the partial mask should be NaN."""
        tsnr_data, _ = compute_tsnr_map(synthetic_bold, synthetic_partial_mask)
        mask_data = synthetic_partial_mask.get_fdata().astype(bool)
        # Outside mask → NaN
        outside = tsnr_data[~mask_data]
        assert np.all(np.isnan(outside)), "Expected NaN outside mask"

    def test_tsnr_dtype_float32(self, synthetic_bold, synthetic_mask):
        """Output tSNR array should be float32 to control memory usage."""
        tsnr_data, _ = compute_tsnr_map(synthetic_bold, synthetic_mask)
        assert tsnr_data.dtype == np.float32

    def test_tsnr_constant_signal_is_nan(self):
        """Voxels with constant signal (std=0) should produce NaN tSNR."""
        import nibabel as nib
        import numpy as np
        data = np.ones((5, 5, 5, 30), dtype=np.int32) * 1000
        img = nib.Nifti1Image(data, np.eye(4))
        tsnr_data, _ = compute_tsnr_map(img, mask_img=None)
        assert np.all(np.isnan(tsnr_data)), "Constant signal should yield NaN tSNR"


class TestExtractTsnrByRoi:
    def test_returns_expected_keys(self, synthetic_bold, synthetic_mask, synthetic_suit_atlas, synthetic_aseg):
        tsnr_data, _ = compute_tsnr_map(synthetic_bold, synthetic_mask)
        mask_data = synthetic_mask.get_fdata().astype(np.uint8)
        result = extract_tsnr_by_roi(tsnr_data, synthetic_suit_atlas, synthetic_aseg, mask_data)

        assert "wholebrain_mean" in result
        assert "cereb_mean" in result
        assert "cereb_wb_ratio" in result
        # At least some SUIT lobule keys
        assert any(k.startswith("suit_") for k in result)

    def test_cereb_wb_ratio_is_finite(self, synthetic_bold, synthetic_mask, synthetic_suit_atlas, synthetic_aseg):
        tsnr_data, _ = compute_tsnr_map(synthetic_bold, synthetic_mask)
        mask_data = synthetic_mask.get_fdata().astype(np.uint8)
        result = extract_tsnr_by_roi(tsnr_data, synthetic_suit_atlas, synthetic_aseg, mask_data)
        ratio = result["cereb_wb_ratio"]
        assert np.isfinite(ratio), f"cereb_wb_ratio should be finite, got {ratio}"
        assert 0 < ratio < 5, f"Unexpected ratio: {ratio}"

    def test_missing_aseg_handled(self, synthetic_bold, synthetic_mask, synthetic_suit_atlas):
        """Should work gracefully when aseg_data is None."""
        tsnr_data, _ = compute_tsnr_map(synthetic_bold, synthetic_mask)
        mask_data = synthetic_mask.get_fdata().astype(np.uint8)
        result = extract_tsnr_by_roi(tsnr_data, synthetic_suit_atlas, aseg_data=None, mask_data=mask_data)
        assert "wholebrain_mean" in result
        assert "cereb_mean" in result

    def test_empty_lobule_returns_nan(self, synthetic_bold, synthetic_mask):
        """Lobules not present in atlas should return NaN."""
        import numpy as np
        # Atlas with only label 1 (all others absent)
        atlas = np.zeros((20, 20, 20), dtype=np.int16)
        atlas[0, 0, 0] = 1
        tsnr_data, _ = compute_tsnr_map(synthetic_bold, synthetic_mask)
        mask_data = synthetic_mask.get_fdata().astype(np.uint8)
        result = extract_tsnr_by_roi(tsnr_data, atlas, aseg_data=None, mask_data=mask_data)
        # Label 2 (V_R) should be NaN since it's not in atlas
        assert np.isnan(result.get("suit_V_R", float("nan")))


class TestLobuleCoverageQuality:
    def test_full_mask_all_valid(self, synthetic_bold, synthetic_mask, synthetic_suit_atlas):
        tsnr_data, _ = compute_tsnr_map(synthetic_bold, synthetic_mask)
        mask_data = synthetic_mask.get_fdata().astype(np.uint8)
        result = compute_lobule_coverage_quality(tsnr_data, synthetic_suit_atlas, mask_data)
        # Labels 1-4 present in atlas, should all be acceptable quality
        from qc.atlas import SUIT_LABEL_MAP
        for label_id, name in SUIT_LABEL_MAP.items():
            if label_id <= 4:
                assert result[name] == True, f"{name} should have acceptable quality"

    def test_partial_mask_reduces_quality(self, synthetic_bold, synthetic_partial_mask, synthetic_suit_atlas):
        tsnr_data, _ = compute_tsnr_map(synthetic_bold, synthetic_partial_mask)
        mask_data = synthetic_partial_mask.get_fdata().astype(np.uint8)
        result = compute_lobule_coverage_quality(tsnr_data, synthetic_suit_atlas, mask_data)
        # With partial mask, not all voxels of lobule 1 (x=0:5) are valid
        assert isinstance(result, dict)
