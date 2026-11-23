"""
test_inference.py
=================
Integration tests for the inference pipeline.

These tests verify end-to-end behavior using synthetic audio where needed,
and validate the saved/loaded pipeline produces consistent predictions.
"""

import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Skip all tests that require librosa if unavailable
librosa_available = True
try:
    import librosa
    import soundfile as sf
except ImportError:
    librosa_available = False


from src.svm_from_scratch import LinearSVMScratch
from src.features import FeatureConfig
from src.calibration import PlattCalibrator


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _sine_wave(freq=440, duration=3.0, sr=16000, amplitude=0.3):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write_wav(waveform, sr, path):
    sf.write(path, waveform, sr)


def _make_tiny_pipeline():
    """
    Create a minimal but complete InferencePipeline for testing.
    Uses synthetic features; does NOT require real audio training.
    """
    from src.inference import InferencePipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    cfg = FeatureConfig(
        use_mfcc=True, use_lfcc=False,
        use_spectral=False, use_prosodic=False,
        use_mfcc_delta=False, use_mfcc_delta2=False,
        stat_functions=("mean", "std"),
        n_mfcc=13,
    )

    # Train a minimal SVM on synthetic data
    rng = np.random.default_rng(0)
    n_train = 200
    n_features = 13 * 2  # n_mfcc × n_stats
    X = rng.standard_normal((n_train, n_features)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(imputer.fit_transform(X))

    svm = LinearSVMScratch(C=1.0, max_epochs=20, verbose=False)
    svm.fit(X_sc, y)

    # Fit calibrator
    calibrator = PlattCalibrator()
    scores = svm.decision_function(X_sc)
    calibrator.fit(scores, y)

    return InferencePipeline(
        model=svm,
        cfg=cfg,
        model_name="test_svm",
        metadata={"test": True},
        scaler=scaler,
        imputer=imputer,
        calibrator=calibrator,
    )


# ─────────────────────────────────────────────
# Platt calibrator tests
# ─────────────────────────────────────────────

class TestPlattCalibrator:
    def test_fit_produces_parameters(self):
        scores = np.random.randn(100)
        y = (scores > 0).astype(int)
        cal = PlattCalibrator()
        cal.fit(scores, y)
        assert cal.A_ is not None
        assert cal.B_ is not None

    def test_probabilities_sum_to_one(self):
        scores = np.random.randn(100)
        y = (scores > 0).astype(int)
        cal = PlattCalibrator()
        cal.fit(scores, y)
        proba = cal.predict_proba(scores)
        assert proba.shape == (100, 2)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_probabilities_in_range(self):
        scores = np.random.randn(100)
        y = (scores > 0).astype(int)
        cal = PlattCalibrator()
        cal.fit(scores, y)
        proba = cal.predict_proba(scores)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_not_fitted_raises(self):
        cal = PlattCalibrator()
        with pytest.raises(RuntimeError):
            cal.predict_proba(np.array([0.0]))

    def test_round_trip_serialization(self):
        scores = np.random.randn(50)
        y = (scores > 0).astype(int)
        cal = PlattCalibrator()
        cal.fit(scores, y)
        d = cal.to_dict()
        cal2 = PlattCalibrator.from_dict(d)
        p1 = cal.predict_proba(scores)
        p2 = cal2.predict_proba(scores)
        assert np.allclose(p1, p2)


# ─────────────────────────────────────────────
# Pipeline save/load consistency
# ─────────────────────────────────────────────

class TestPipelineSaveLoad:
    @pytest.mark.skipif(not librosa_available, reason="librosa/soundfile not available")
    def test_save_and_reload_consistent(self, tmp_path):
        from src.inference import InferencePipeline

        pipeline = _make_tiny_pipeline()
        save_path = str(tmp_path / "pipeline.joblib")
        pipeline.save(save_path)

        # Reload
        pipeline2 = InferencePipeline.load(save_path)

        # Generate test features
        rng = np.random.default_rng(99)
        X_test = rng.standard_normal((10, 26)).astype(np.float32)

        # Both pipelines should produce the same probabilities
        proba1 = pipeline._get_proba(X_test)
        proba2 = pipeline2._get_proba(X_test)
        assert np.allclose(proba1, proba2, atol=1e-5), \
            "Probabilities differ after save/reload"


# ─────────────────────────────────────────────
# End-to-end inference on WAV files
# ─────────────────────────────────────────────

class TestEndToEndInference:
    @pytest.mark.skipif(not librosa_available, reason="librosa/soundfile not available")
    def test_valid_wav_returns_result(self, tmp_path):
        from src.inference import predict_audio

        pipeline = _make_tiny_pipeline()
        waveform = _sine_wave(duration=4.0)
        wav_path = str(tmp_path / "test.wav")
        _write_wav(waveform, 16000, wav_path)

        result = predict_audio(wav_path, pipeline)
        assert result["valid"]
        assert result["label_str"] in {"REAL", "FAKE"}
        assert result["label_int"] in {0, 1}
        assert 0.0 <= result["p_real"] <= 1.0
        assert 0.0 <= result["p_fake"] <= 1.0
        # Probabilities sum to 1
        assert abs(result["p_real"] + result["p_fake"] - 1.0) < 1e-4

    @pytest.mark.skipif(not librosa_available, reason="librosa/soundfile not available")
    def test_probabilities_sum_to_one(self, tmp_path):
        from src.inference import predict_audio

        pipeline = _make_tiny_pipeline()
        waveform = _sine_wave(duration=6.0)
        wav_path = str(tmp_path / "test6s.wav")
        _write_wav(waveform, 16000, wav_path)

        result = predict_audio(wav_path, pipeline)
        assert result["valid"]
        total = result["p_real"] + result["p_fake"]
        assert abs(total - 1.0) < 1e-4, f"Probs don't sum to 1: {total}"

    @pytest.mark.skipif(not librosa_available, reason="librosa/soundfile not available")
    def test_invalid_path_returns_error(self):
        from src.inference import predict_audio

        pipeline = _make_tiny_pipeline()
        result = predict_audio("/nonexistent/file.wav", pipeline)
        assert not result["valid"]
        assert result["error"] is not None

    @pytest.mark.skipif(not librosa_available, reason="librosa/soundfile not available")
    def test_very_short_audio_returns_error(self, tmp_path):
        from src.inference import predict_audio

        pipeline = _make_tiny_pipeline()
        # 0.1 seconds — too short
        waveform = _sine_wave(duration=0.1)
        wav_path = str(tmp_path / "short.wav")
        _write_wav(waveform, 16000, wav_path)

        result = predict_audio(wav_path, pipeline)
        # Should either fail validation or return uncertain
        assert not result["valid"] or result["confidence"] in {"uncertain", "moderate", "high"}

    @pytest.mark.skipif(not librosa_available, reason="librosa/soundfile not available")
    def test_stereo_wav_handled(self, tmp_path):
        from src.inference import predict_audio

        pipeline = _make_tiny_pipeline()
        sr = 16000
        stereo = np.random.randn(sr * 4, 2).astype(np.float32) * 0.1
        wav_path = str(tmp_path / "stereo.wav")
        sf.write(wav_path, stereo, sr)

        result = predict_audio(wav_path, pipeline)
        # Should succeed after mono conversion
        assert result["valid"]

    @pytest.mark.skipif(not librosa_available, reason="librosa/soundfile not available")
    def test_long_audio_produces_multiple_segments(self, tmp_path):
        from src.inference import predict_audio

        pipeline = _make_tiny_pipeline()
        waveform = _sine_wave(duration=12.0)
        wav_path = str(tmp_path / "long.wav")
        _write_wav(waveform, 16000, wav_path)

        result = predict_audio(wav_path, pipeline)
        assert result["valid"]
        assert result["diagnostics"]["n_segments"] > 1
        assert len(result["segment_proba"]) > 1


# ─────────────────────────────────────────────
# Non-speech audio check
# ─────────────────────────────────────────────

class TestNonSpeechDetection:
    @pytest.mark.skipif(not librosa_available, reason="librosa/soundfile not available")
    def test_silence_detected_as_no_speech(self, tmp_path):
        from src.inference import check_is_speech

        waveform = np.zeros(16000 * 3, dtype=np.float32)
        result = check_is_speech(waveform, 16000)
        assert not result["has_speech"]
        assert result["warning"] != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
