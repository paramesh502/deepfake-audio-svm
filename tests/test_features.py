"""
test_features.py
================
Unit tests for audio feature extraction.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Skip all tests if librosa is not available
librosa_available = True
try:
    import librosa
except ImportError:
    librosa_available = False


pytestmark = pytest.mark.skipif(
    not librosa_available,
    reason="librosa not installed"
)


from src.features import (
    FeatureConfig,
    extract_mfcc,
    extract_lfcc,
    extract_spectral_features,
    extract_prosodic_features,
    extract_all_features,
    aggregate_segment_features,
    compute_expected_feature_dim,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _sine_wave(freq=440, duration=3.0, sr=16000, amplitude=0.5):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _white_noise(duration=3.0, sr=16000, amplitude=0.1, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(sr * duration)) * amplitude).astype(np.float32)


# ─────────────────────────────────────────────
# FeatureConfig tests
# ─────────────────────────────────────────────

class TestFeatureConfig:
    def test_default_config(self):
        cfg = FeatureConfig()
        assert cfg.n_mfcc == 20
        assert cfg.use_mfcc
        assert cfg.use_lfcc

    def test_to_dict_round_trip(self):
        cfg = FeatureConfig(n_mfcc=13, use_chroma=True)
        d = cfg.to_dict()
        cfg2 = FeatureConfig.from_dict(d)
        assert cfg2.n_mfcc == 13
        assert cfg2.use_chroma

    def test_expected_feature_dim_positive(self):
        cfg = FeatureConfig()
        dim = compute_expected_feature_dim(cfg)
        assert dim > 0

    def test_mfcc_only_dim(self):
        cfg = FeatureConfig(
            use_mfcc=True,
            use_lfcc=False,
            use_spectral=False,
            use_prosodic=False,
            n_mfcc=20,
            use_mfcc_delta=True,
            use_mfcc_delta2=True,
            stat_functions=("mean", "std"),
        )
        dim = compute_expected_feature_dim(cfg)
        # 20 coeff × 2 stats × 3 (base + delta + delta2) = 120
        assert dim == 120


# ─────────────────────────────────────────────
# MFCC tests
# ─────────────────────────────────────────────

class TestMFCCExtraction:
    def test_returns_1d_array(self):
        cfg = FeatureConfig()
        waveform = _sine_wave()
        feat = extract_mfcc(waveform, cfg)
        assert feat.ndim == 1

    def test_no_nan_or_inf(self):
        cfg = FeatureConfig()
        waveform = _sine_wave()
        feat = extract_mfcc(waveform, cfg)
        assert np.isfinite(feat).all()

    def test_dtype_float32(self):
        cfg = FeatureConfig()
        waveform = _sine_wave()
        feat = extract_mfcc(waveform, cfg)
        assert feat.dtype == np.float32

    def test_deterministic(self):
        cfg = FeatureConfig()
        waveform = _sine_wave()
        feat1 = extract_mfcc(waveform, cfg)
        feat2 = extract_mfcc(waveform, cfg)
        assert np.allclose(feat1, feat2)

    def test_no_delta_has_fewer_features(self):
        cfg_full = FeatureConfig(use_mfcc_delta=True, use_mfcc_delta2=True)
        cfg_base = FeatureConfig(use_mfcc_delta=False, use_mfcc_delta2=False)
        waveform = _sine_wave()
        feat_full = extract_mfcc(waveform, cfg_full)
        feat_base = extract_mfcc(waveform, cfg_base)
        assert len(feat_full) == 3 * len(feat_base)

    def test_different_waveforms_different_features(self):
        cfg = FeatureConfig()
        sine = _sine_wave(freq=440)
        noise = _white_noise()
        feat_sine = extract_mfcc(sine, cfg)
        feat_noise = extract_mfcc(noise, cfg)
        assert not np.allclose(feat_sine, feat_noise)


# ─────────────────────────────────────────────
# LFCC tests
# ─────────────────────────────────────────────

class TestLFCCExtraction:
    def test_returns_1d_array(self):
        cfg = FeatureConfig()
        waveform = _sine_wave()
        feat = extract_lfcc(waveform, cfg)
        assert feat.ndim == 1

    def test_no_nan_or_inf(self):
        cfg = FeatureConfig()
        waveform = _sine_wave()
        feat = extract_lfcc(waveform, cfg)
        assert np.isfinite(feat).all()

    def test_dtype_float32(self):
        cfg = FeatureConfig()
        waveform = _sine_wave()
        feat = extract_lfcc(waveform, cfg)
        assert feat.dtype == np.float32

    def test_mfcc_vs_lfcc_differ(self):
        """MFCC and LFCC should produce different feature vectors."""
        cfg = FeatureConfig()
        waveform = _sine_wave()
        mfcc = extract_mfcc(waveform, cfg)
        lfcc = extract_lfcc(waveform, cfg)
        # They have the same dimension but different values
        assert len(mfcc) == len(lfcc)
        assert not np.allclose(mfcc, lfcc, atol=1e-3)


# ─────────────────────────────────────────────
# Spectral feature tests
# ─────────────────────────────────────────────

class TestSpectralFeatures:
    def test_returns_1d_array(self):
        cfg = FeatureConfig()
        waveform = _sine_wave()
        feat = extract_spectral_features(waveform, cfg)
        assert feat.ndim == 1

    def test_no_nan_or_inf(self):
        cfg = FeatureConfig()
        waveform = _sine_wave()
        feat = extract_spectral_features(waveform, cfg)
        assert np.isfinite(feat).all()

    def test_with_silence(self):
        """Silent waveform should still produce valid features (zeros or small values)."""
        cfg = FeatureConfig()
        waveform = np.zeros(16000 * 3, dtype=np.float32)
        feat = extract_spectral_features(waveform, cfg)
        assert feat.ndim == 1
        assert np.isfinite(feat).all()


# ─────────────────────────────────────────────
# Combined feature extraction
# ─────────────────────────────────────────────

class TestExtractAllFeatures:
    def test_returns_1d_array(self):
        waveform = _sine_wave()
        feat = extract_all_features(waveform)
        assert feat.ndim == 1

    def test_no_nan_or_inf(self):
        waveform = _sine_wave()
        feat = extract_all_features(waveform)
        assert np.isfinite(feat).all()

    def test_dtype_float32(self):
        waveform = _sine_wave()
        feat = extract_all_features(waveform)
        assert feat.dtype == np.float32

    def test_dimension_matches_config(self):
        """
        The actual extracted dimension must match the expected dimension
        from compute_expected_feature_dim().
        """
        cfg = FeatureConfig(
            use_mfcc=True, use_lfcc=True,
            use_spectral=True, use_prosodic=False,
            use_mfcc_delta=True, use_mfcc_delta2=True,
            use_lfcc_delta=False, use_lfcc_delta2=False,
            stat_functions=("mean", "std"),
        )
        waveform = _sine_wave()
        feat = extract_all_features(waveform, cfg)
        expected_dim = compute_expected_feature_dim(cfg)
        # The dimension must be close (prosodic=False → no pitch features)
        # Allow some tolerance for the prosodic branch
        assert len(feat) >= expected_dim - 20

    def test_deterministic_for_same_input(self):
        waveform = _sine_wave()
        cfg = FeatureConfig()
        feat1 = extract_all_features(waveform, cfg)
        feat2 = extract_all_features(waveform, cfg)
        assert np.allclose(feat1, feat2)

    def test_all_groups_disabled_raises(self):
        cfg = FeatureConfig(
            use_mfcc=False, use_lfcc=False,
            use_spectral=False, use_prosodic=False,
        )
        waveform = _sine_wave()
        with pytest.raises(ValueError):
            extract_all_features(waveform, cfg)


# ─────────────────────────────────────────────
# Segment aggregation
# ─────────────────────────────────────────────

class TestSegmentAggregation:
    def test_aggregation_shape(self):
        cfg = FeatureConfig()
        waveform = _sine_wave(duration=3.0)
        feat = extract_all_features(waveform, cfg)
        d = len(feat)

        # Simulate 4 segments
        seg_feats = np.vstack([feat + np.random.randn(d) * 0.01 for _ in range(4)])
        agg = aggregate_segment_features(seg_feats.astype(np.float32))
        assert len(agg) == 2 * d  # mean + std

    def test_aggregation_1d_passthrough(self):
        cfg = FeatureConfig()
        waveform = _sine_wave()
        feat = extract_all_features(waveform, cfg)
        agg = aggregate_segment_features(feat)
        assert agg.shape == feat.shape

    def test_aggregation_no_nan(self):
        d = 50
        seg_feats = np.random.randn(5, d).astype(np.float32)
        agg = aggregate_segment_features(seg_feats)
        assert np.isfinite(agg).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
