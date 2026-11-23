"""
preprocessing.py
================
Audio loading, validation, conversion, resampling, segmentation,
and Speech Activity Detection (SAD) utilities.

All operations here are shared between training-time batch processing
and runtime inference, so the two paths produce identical features.
"""

import os
import warnings
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

# Optional heavy imports — degrade gracefully so unit tests that only
# use the numeric helpers can still run without audio libraries.
try:
    import librosa
    import librosa.effects
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False
    warnings.warn("librosa not found. Audio loading will be unavailable.")

try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
TARGET_SR: int = 16_000          # Hz — standard for speech
TARGET_CHANNELS: int = 1         # mono
SEGMENT_DURATION: float = 3.0    # seconds
SEGMENT_HOP: float = 1.5         # seconds (50 % overlap)
MIN_DURATION: float = 0.5        # seconds — reject shorter clips
MAX_DURATION: float = 300.0      # seconds — guard memory
MAX_FILE_MB: float = 200.0       # MB
CLIPPING_THRESHOLD: float = 0.99 # fraction of samples at ±1 that counts as clipping
MIN_SPEECH_RATIO: float = 0.10   # at least 10 % voiced frames required
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

# ─────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────

def validate_file(path: str) -> Dict[str, Any]:
    """
    Lightweight file-system validation before attempting to decode.

    Returns a dict with keys:
        valid   : bool
        reason  : str  (empty when valid)
        size_mb : float
    """
    p = Path(path)
    result = {"valid": False, "reason": "", "size_mb": 0.0}

    if not p.exists():
        result["reason"] = f"File not found: {path}"
        return result

    if not p.is_file():
        result["reason"] = f"Path is not a file: {path}"
        return result

    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        result["reason"] = (
            f"Unsupported extension '{p.suffix}'. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )
        return result

    size_mb = p.stat().st_size / (1024 ** 2)
    result["size_mb"] = size_mb

    if size_mb > MAX_FILE_MB:
        result["reason"] = (
            f"File too large ({size_mb:.1f} MB). "
            f"Maximum allowed: {MAX_FILE_MB} MB."
        )
        return result

    result["valid"] = True
    return result


# ─────────────────────────────────────────────
# Core audio loading
# ─────────────────────────────────────────────

def load_audio(
    path: str,
    target_sr: int = TARGET_SR,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Decode an audio file and return (waveform, sample_rate).

    - Resamples to target_sr.
    - Converts to mono by averaging channels.
    - Returns float32 waveform in [-1, 1].

    Raises
    ------
    RuntimeError
        When librosa is unavailable or decoding fails.
    """
    if not _LIBROSA_AVAILABLE:
        raise RuntimeError(
            "librosa is required for audio loading. "
            "Install it with: pip install librosa"
        )

    try:
        # librosa handles WAV, FLAC, MP3, OGG, M4A via soundfile/audioread
        waveform, sr = librosa.load(path, sr=target_sr, mono=mono)
    except Exception as exc:
        raise RuntimeError(f"Failed to load '{path}': {exc}") from exc

    waveform = waveform.astype(np.float32)
    return waveform, sr


# ─────────────────────────────────────────────
# Quality checks
# ─────────────────────────────────────────────

def check_duration(waveform: np.ndarray, sr: int) -> Dict[str, Any]:
    """Check whether the clip is too short or too long."""
    duration = len(waveform) / sr
    result = {"duration_s": duration, "valid": True, "reason": ""}
    if duration < MIN_DURATION:
        result["valid"] = False
        result["reason"] = (
            f"Recording too short ({duration:.2f} s). "
            f"Minimum required: {MIN_DURATION} s."
        )
    elif duration > MAX_DURATION:
        result["valid"] = False
        result["reason"] = (
            f"Recording too long ({duration:.0f} s). "
            f"Maximum allowed: {MAX_DURATION} s."
        )
    return result


def check_clipping(waveform: np.ndarray, threshold: float = CLIPPING_THRESHOLD) -> Dict[str, Any]:
    """
    Estimate the fraction of samples at or near ±1 to detect hard clipping.

    A synthetic vocoder artefact and a clipped microphone recording are
    acoustically very different; we warn but do not reject clipped audio,
    since clipping features may themselves aid classification.
    """
    n_clipped = np.sum(np.abs(waveform) >= threshold)
    clip_frac = n_clipped / max(len(waveform), 1)
    is_clipped = clip_frac > 0.01  # more than 1 % clipped samples
    return {
        "clip_fraction": float(clip_frac),
        "is_clipped": bool(is_clipped),
        "warning": (
            f"Severe clipping detected ({clip_frac*100:.1f}% of samples). "
            "Classification may be less reliable."
            if is_clipped else ""
        ),
    }


def check_silence(waveform: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> Dict[str, Any]:
    """
    Lightweight energy-based VAD to estimate the fraction of voiced frames.
    Uses RMS energy per frame; frames below 1% of max RMS are silent.
    """
    if not _LIBROSA_AVAILABLE:
        # Fallback: simple RMS check on whole signal
        rms = float(np.sqrt(np.mean(waveform ** 2)))
        return {"speech_ratio": float(rms > 0.01), "has_speech": rms > 0.01, "warning": ""}

    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
    threshold = 0.01 * rms.max() if rms.max() > 0 else 1e-9
    n_speech = np.sum(rms > threshold)
    speech_ratio = n_speech / max(len(rms), 1)
    has_speech = speech_ratio >= MIN_SPEECH_RATIO
    return {
        "speech_ratio": float(speech_ratio),
        "has_speech": bool(has_speech),
        "warning": (
            f"Insufficient speech detected ({speech_ratio*100:.1f}% voiced frames). "
            "Result may be unreliable."
            if not has_speech else ""
        ),
    }


# ─────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────

def normalize_waveform(waveform: np.ndarray, method: str = "peak") -> np.ndarray:
    """
    Amplitude normalisation.

    method="peak"  : divide by peak absolute value → max amplitude = 1
    method="rms"   : scale to target RMS = 0.1
    """
    if method == "peak":
        peak = np.abs(waveform).max()
        if peak > 1e-9:
            waveform = waveform / peak
    elif method == "rms":
        target_rms = 0.1
        rms = float(np.sqrt(np.mean(waveform ** 2)))
        if rms > 1e-9:
            waveform = waveform * (target_rms / rms)
            waveform = np.clip(waveform, -1.0, 1.0)
    else:
        raise ValueError(f"Unknown normalisation method: {method}")
    return waveform


# ─────────────────────────────────────────────
# Segmentation
# ─────────────────────────────────────────────

def segment_waveform(
    waveform: np.ndarray,
    sr: int,
    segment_duration: float = SEGMENT_DURATION,
    hop: float = SEGMENT_HOP,
) -> List[np.ndarray]:
    """
    Split a waveform into overlapping fixed-length segments.

    Returns a list of numpy arrays, each of length sr * segment_duration.
    If the input is shorter than one segment, the single (zero-padded)
    segment is returned so the pipeline always has at least one segment
    to process.
    """
    seg_len = int(sr * segment_duration)
    hop_len = int(sr * hop)
    n_samples = len(waveform)

    segments = []
    start = 0
    while start + seg_len <= n_samples:
        segments.append(waveform[start: start + seg_len].copy())
        start += hop_len

    # Always include at least the first segment
    if not segments:
        padded = np.zeros(seg_len, dtype=np.float32)
        padded[: min(n_samples, seg_len)] = waveform[: min(n_samples, seg_len)]
        segments.append(padded)

    return segments


# ─────────────────────────────────────────────
# High-level pipeline
# ─────────────────────────────────────────────

def preprocess_audio_file(
    path: str,
    target_sr: int = TARGET_SR,
    normalize: bool = True,
    normalization_method: str = "peak",
    segment: bool = True,
    segment_duration: float = SEGMENT_DURATION,
    segment_hop: float = SEGMENT_HOP,
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline for a single audio file.

    Returns
    -------
    dict with keys:
        valid       : bool
        warnings    : list[str]
        errors      : list[str]
        waveform    : np.ndarray | None   (full, normalised waveform)
        segments    : list[np.ndarray]    (list of fixed-length segments)
        sr          : int
        duration_s  : float
        diagnostics : dict
    """
    result: Dict[str, Any] = {
        "valid": False,
        "warnings": [],
        "errors": [],
        "waveform": None,
        "segments": [],
        "sr": target_sr,
        "duration_s": 0.0,
        "diagnostics": {},
    }

    # 1. File-system validation
    fs_check = validate_file(path)
    if not fs_check["valid"]:
        result["errors"].append(fs_check["reason"])
        return result

    # 2. Decode
    try:
        waveform, sr = load_audio(path, target_sr=target_sr, mono=True)
    except RuntimeError as exc:
        result["errors"].append(str(exc))
        return result

    # 3. Duration check
    dur_check = check_duration(waveform, sr)
    result["duration_s"] = dur_check["duration_s"]
    if not dur_check["valid"]:
        result["errors"].append(dur_check["reason"])
        return result

    # 4. Clipping check (warning only)
    clip_check = check_clipping(waveform)
    result["diagnostics"]["clipping"] = clip_check
    if clip_check["warning"]:
        result["warnings"].append(clip_check["warning"])

    # 5. Speech activity check (warning only)
    sad_check = check_silence(waveform, sr)
    result["diagnostics"]["speech_activity"] = sad_check
    if sad_check["warning"]:
        result["warnings"].append(sad_check["warning"])

    # 6. Normalise
    if normalize:
        waveform = normalize_waveform(waveform, method=normalization_method)

    # 7. Segment
    if segment:
        segs = segment_waveform(waveform, sr, segment_duration, segment_hop)
    else:
        segs = [waveform]

    result["valid"] = True
    result["waveform"] = waveform
    result["segments"] = segs
    result["sr"] = sr
    return result


# ─────────────────────────────────────────────
# Batch helper used in training
# ─────────────────────────────────────────────

def load_dataset_files(
    file_list: List[str],
    labels: Optional[List[int]] = None,
    target_sr: int = TARGET_SR,
    max_files: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load a list of audio files and return (waveforms, labels, valid_paths).

    Returns only the files that were successfully loaded.
    """
    waveforms, valid_labels, valid_paths = [], [], []

    if labels is None:
        labels = [0] * len(file_list)

    n = min(len(file_list), max_files) if max_files else len(file_list)

    for i in range(n):
        path = file_list[i]
        label = labels[i]
        try:
            info = preprocess_audio_file(path, target_sr=target_sr, segment=False)
            if info["valid"] and info["waveform"] is not None:
                waveforms.append(info["waveform"])
                valid_labels.append(label)
                valid_paths.append(path)
                if verbose and (i + 1) % 500 == 0:
                    logger.info(f"Loaded {i+1}/{n} files")
            else:
                if verbose:
                    logger.warning(f"Skipped {path}: {info['errors']}")
        except Exception as exc:
            if verbose:
                logger.warning(f"Error loading {path}: {exc}")

    logger.info(f"Successfully loaded {len(waveforms)}/{n} files.")
    return waveforms, valid_labels, valid_paths
