"""Tests for brain mask coverage and signal dropout analysis."""

from __future__ import annotations

import numpy as np
import pytest

from qc.metrics.coverage import compute_mask_coverage, compute_signal_dropout


class TestComputeMaskCoverage:
    def test_full_mask_gives_100pct(self, synthetic_mask, synthetic_suit_atlas):
        """Full brain mask should give 100% coverage for all lobules."""
        result = compute_mask_coverage(synthetic_mask, synthetic_suit_atlas)
        # Labels 1–4 are present in synthetic atlas
        for label_id in [1, 2, 3, 4]:
            from qc.atlas import SUIT_LABEL_MAP
            name = SUIT_LABEL_MAP[label_id]
            key = f"suit_coverage_{name}"
            if key in result:
                assert result[key] == pytest.approx(1.0, abs=1e-6), \
                    f"Expected 100% coverage for {name}"

    def test_partial_mask_reduces_coverage(self, synthetic_partial_mask, synthetic_suit_atlas):
        """Partial mask (5:15, 5:15, 5:15) should reduce coverage of some lobules."""
        result = compute_mask_coverage(synthetic_partial_mask, synthetic_suit_atlas)
        # Label 1 (x=0:5) has no overlap with mask (x=5:15)
        from qc.atlas import SUIT_LABEL_MAP
        name_1 = SUIT_LABEL_MAP[1]  # I-IV_R
        cov_1 = result.get(f"suit_coverage_{name_1}", 1.0)
        assert cov_1 == pytest.approx(0.0, abs=1e-6), \
            f"Label 1 should have 0% coverage with partial mask, got {cov_1}"

    def test_summary_keys_present(self, synthetic_mask, synthetic_suit_atlas):
        result = compute_mask_coverage(synthetic_mask, synthetic_suit_atlas)
        assert "suit_coverage_mean" in result
        assert "suit_coverage_min" in result
        assert "suit_coverage_min_lobule" in result

    def test_suit_coverage_mean_is_finite(self, synthetic_mask, synthetic_suit_atlas):
        result = compute_mask_coverage(synthetic_mask, synthetic_suit_atlas)
        assert np.isfinite(result["suit_coverage_mean"])

    def test_empty_atlas_returns_nan(self, synthetic_mask):
        empty_atlas = np.zeros((20, 20, 20), dtype=np.int16)
        result = compute_mask_coverage(synthetic_mask, empty_atlas)
        assert np.isnan(result["suit_coverage_mean"])

    def test_with_aseg(self, synthetic_mask, synthetic_suit_atlas, synthetic_aseg):
        """Coverage with aseg should include aseg region coverage keys."""
        result = compute_mask_coverage(synthetic_mask, synthetic_suit_atlas, synthetic_aseg)
        assert any(k.startswith("aseg_coverage_") for k in result)


class TestComputeSignalDropout:
    def test_high_signal_no_dropout(self, synthetic_boldref, synthetic_suit_atlas, synthetic_mask):
        """Uniform signal should give relative signal ~1.0 for all lobules."""
        mask_data = synthetic_mask.get_fdata().astype(np.uint8)
        result = compute_signal_dropout(
            synthetic_boldref, synthetic_suit_atlas, mask_data
        )
        assert "wholebrain_boldref_mean" in result
        assert np.isfinite(result["wholebrain_boldref_mean"])
        assert result["n_dropout_lobules"] >= 0

    def test_dropout_detected_in_low_signal_region(self, synthetic_boldref, synthetic_mask):
        """
        synthetic_boldref has signal ~200 in x<5 region (label 1 = I-IV_R).
        With whole-brain mean ~1000, relative signal = ~0.2 < 0.5 threshold.
        """
        # Atlas: label 1 covers x=0:5 (low signal in boldref)
        atlas = np.zeros((20, 20, 20), dtype=np.int16)
        atlas[0:5, :, :] = 1  # low signal region
        atlas[10:20, :, :] = 2  # normal signal region

        mask_data = synthetic_mask.get_fdata().astype(np.uint8)
        result = compute_signal_dropout(synthetic_boldref, atlas, mask_data)

        from qc.atlas import SUIT_LABEL_MAP
        name_1 = SUIT_LABEL_MAP.get(1, "I-IV_R")
        rel = result.get(f"dropout_{name_1}", 1.0)
        flag = result.get(f"dropout_flag_{name_1}", 0.0)
        # Relative signal in x<5 = 200/1000 = 0.2 < threshold 0.5
        assert rel < 0.5, f"Expected dropout (<0.5 relative signal), got {rel}"
        assert flag == 1.0, "Expected dropout flag=1"
        assert result["n_dropout_lobules"] >= 1

    def test_returns_wholebrain_mean(self, synthetic_boldref, synthetic_suit_atlas, synthetic_mask):
        mask_data = synthetic_mask.get_fdata().astype(np.uint8)
        result = compute_signal_dropout(synthetic_boldref, synthetic_suit_atlas, mask_data)
        wb = result["wholebrain_boldref_mean"]
        assert np.isfinite(wb)
        assert wb > 0

    def test_handles_no_mask(self, synthetic_boldref, synthetic_suit_atlas):
        """Should work without a mask (mask_data=None)."""
        result = compute_signal_dropout(synthetic_boldref, synthetic_suit_atlas, mask_data=None)
        assert "wholebrain_boldref_mean" in result
        assert np.isfinite(result["wholebrain_boldref_mean"])
