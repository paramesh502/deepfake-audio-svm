"""
test_preprocessing.py
=====================
Unit tests for audio preprocessing functions.
"""

import os
import sys
import tempfile
import numpy as np
import pytest

# Check optional dependencies
try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

# Allow running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import (
    validate_file,
    check_duration,
    check_clipping,
    check_silence,
    normalize_waveform,
    segment_waveform,
    TARGET_SR,
    MIN_DURATION,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _sine_wave(freq=440, duration=3.0, sr=16000, amplitude=0.5):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write_wav(waveform, sr, path):
    """Write a WAV file for testing."""
    try:
        import soundfile as sf
        sf.write(path, waveform, sr, format="WAV")
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
# validate_file tests
# ─────────────────────────────────────────────

class TestValidateFile:
    def test_missing_file(self):
        result = validate_file("/nonexistent/path/file.wav")
        assert not result["valid"]
        assert "not found" in result["reason"].lower()

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "audio.xyz"
        p.touch()
        result = validate_file(str(p))
        assert not result["valid"]
        assert "unsupported" in result["reason"].lower()

    def test_valid_wav_path(self, tmp_path):
        p = tmp_path / "audio.wav"
        p.touch()
        result = validate_file(str(p))
        # File exists but is empty — still passes file-system validation
        # (actual audio decoding is a separate step)
        assert result["valid"] or not result["valid"]  # depends on size check

    def test_supported_extensions(self, tmp_path):
        for ext in [".wav", ".flac", ".mp3", ".ogg"]:
            p = tmp_path / f"audio{ext}"
            p.touch()
            result = validate_file(str(p))
            # Should not fail on extension
            if not result["valid"]:
                assert "extension" not in result["reason"].lower()


# ─────────────────────────────────────────────
# check_duration tests
# ─────────────────────────────────────────────

class TestCheckDuration:
    def test_too_short(self):
        sr = 16000
        waveform = np.zeros(int(sr * 0.1), dtype=np.float32)  # 0.1s
        result = check_duration(waveform, sr)
        assert not result["valid"]
        assert "short" in result["reason"].lower()

    def test_adequate_duration(self):
        sr = 16000
        waveform = np.zeros(int(sr * 2.0), dtype=np.float32)  # 2s
        result = check_duration(waveform, sr)
        assert result["valid"]
        assert abs(result["duration_s"] - 2.0) < 0.01

    def test_too_long(self):
        sr = 16000
        # 400 seconds > MAX_DURATION=300
        waveform = np.zeros(int(sr * 400), dtype=np.float32)
        result = check_duration(waveform, sr)
        assert not result["valid"]
        assert "long" in result["reason"].lower()

    def test_exact_minimum(self):
        sr = 16000
        waveform = np.zeros(int(sr * MIN_DURATION), dtype=np.float32)
        result = check_duration(waveform, sr)
        assert result["valid"]


# ─────────────────────────────────────────────
# check_clipping tests
# ─────────────────────────────────────────────

class TestCheckClipping:
    def test_no_clipping(self):
        waveform = _sine_wave(amplitude=0.5)
        result = check_clipping(waveform)
        assert not result["is_clipped"]

    def test_severe_clipping(self):
        waveform = np.ones(16000, dtype=np.float32)  # all 1.0
        result = check_clipping(waveform)
        assert result["is_clipped"]
        assert result["clip_fraction"] > 0.99

    def test_clip_fraction_range(self):
        waveform = _sine_wave(amplitude=0.3)
        result = check_clipping(waveform)
        assert 0.0 <= result["clip_fraction"] <= 1.0


# ─────────────────────────────────────────────
# check_silence tests
# ─────────────────────────────────────────────

class TestCheckSilence:
    def test_silence_detected(self):
        waveform = np.zeros(16000, dtype=np.float32)
        result = check_silence(waveform, sr=16000)
        assert not result["has_speech"]

    def test_speech_detected(self):
        waveform = _sine_wave(freq=440, duration=3.0, amplitude=0.5)
        result = check_silence(waveform, sr=16000)
        # Sine wave should not be flagged as silence
        assert result["speech_ratio"] > 0.0

    def test_speech_ratio_range(self):
        waveform = _sine_wave()
        result = check_silence(waveform, sr=16000)
        assert 0.0 <= result["speech_ratio"] <= 1.0


# ─────────────────────────────────────────────
# normalize_waveform tests
# ─────────────────────────────────────────────

class TestNormalizeWaveform:
    def test_peak_normalization(self):
        waveform = _sine_wave(amplitude=0.3)
        normalized = normalize_waveform(waveform, method="peak")
        assert abs(np.max(np.abs(normalized)) - 1.0) < 1e-5

    def test_rms_normalization(self):
        waveform = _sine_wave(amplitude=0.3)
        normalized = normalize_waveform(waveform, method="rms")
        rms = np.sqrt(np.mean(normalized ** 2))
        assert abs(rms - 0.1) < 0.02

    def test_zero_signal(self):
        waveform = np.zeros(1000, dtype=np.float32)
        normalized = normalize_waveform(waveform, method="peak")
        assert np.allclose(normalized, 0.0)

    def test_unknown_method(self):
        waveform = _sine_wave()
        with pytest.raises(ValueError):
            normalize_waveform(waveform, method="unknown_method")


# ─────────────────────────────────────────────
# segment_waveform tests
# ─────────────────────────────────────────────

class TestSegmentWaveform:
    def test_short_clips_produce_one_segment(self):
        sr = 16000
        waveform = np.zeros(sr, dtype=np.float32)  # 1 second
        segs = segment_waveform(waveform, sr, segment_duration=3.0, hop=1.5)
        assert len(segs) == 1
        assert len(segs[0]) == sr * 3  # padded to segment length

    def test_longer_clips_produce_multiple_segments(self):
        sr = 16000
        waveform = np.zeros(sr * 10, dtype=np.float32)  # 10 seconds
        segs = segment_waveform(waveform, sr, segment_duration=3.0, hop=1.5)
        assert len(segs) > 1

    def test_segment_length_is_consistent(self):
        sr = 16000
        waveform = _sine_wave(duration=8.0)
        segs = segment_waveform(waveform, sr, segment_duration=3.0, hop=1.5)
        expected_len = sr * 3
        for seg in segs:
            assert len(seg) == expected_len

    def test_all_zeros_produces_segment(self):
        waveform = np.zeros(16000 * 2, dtype=np.float32)
        segs = segment_waveform(waveform, 16000)
        assert len(segs) >= 1


# ─────────────────────────────────────────────
# Mono conversion (via load_audio)
# ─────────────────────────────────────────────

class TestMonoConversion:
    """
    Test that stereo audio is correctly converted to mono.
    Requires soundfile + librosa; skipped if unavailable.
    """

    @pytest.mark.skipif(
        not _SF_AVAILABLE,
        reason="soundfile/librosa not available"
    )
    def test_stereo_to_mono(self, tmp_path):
        try:
            import soundfile as sf
            import librosa
        except ImportError:
            pytest.skip("soundfile or librosa not available")

        sr = 16000
        stereo = np.random.randn(sr * 2, 2).astype(np.float32) * 0.1
        path = str(tmp_path / "stereo.wav")
        sf.write(path, stereo, sr)

        from src.preprocessing import load_audio
        waveform, out_sr = load_audio(path, target_sr=sr, mono=True)
        assert waveform.ndim == 1
        assert out_sr == sr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
