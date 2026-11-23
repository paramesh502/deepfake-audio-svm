"""
features.py
===========
Handcrafted acoustic feature extraction for deepfake speech detection.

Feature groups
--------------
A. MFCC (13–20 coefficients + delta + delta-delta)
B. LFCC (Linear-Frequency Cepstral Coefficients)
C. Spectral features (centroid, bandwidth, rolloff, flatness, contrast, ZCR)
D. Prosodic / temporal features (RMS energy statistics, pitch, voiced ratio)

All functions return fixed-length 1-D float32 numpy arrays.
NaN and Inf values are replaced with 0 before returning.

The final feature vector from extract_all_features() is the concatenation
of whichever groups are enabled in the FeatureConfig.
"""

import warnings
import logging
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from scipy.fftpack import dct
from scipy.signal import lfilter

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False
    warnings.warn("librosa not available. Feature extraction will be limited.")

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Feature configuration
# ─────────────────────────────────────────────

class FeatureConfig:
    """
    Centralised feature configuration.

    Pass an instance to extract_all_features() to reproduce the
    exact feature set used during training.
    """
    def __init__(
        self,
        sr: int = 16_000,
        n_fft: int = 512,
        hop_length: int = 160,            # 10 ms at 16 kHz
        win_length: int = 400,            # 25 ms at 16 kHz
        # MFCC
        use_mfcc: bool = True,
        n_mfcc: int = 20,
        use_mfcc_delta: bool = True,
        use_mfcc_delta2: bool = True,
        # LFCC
        use_lfcc: bool = True,
        n_lfcc: int = 20,
        use_lfcc_delta: bool = True,
        use_lfcc_delta2: bool = True,
        # Spectral
        use_spectral: bool = True,
        n_chroma: int = 12,
        use_chroma: bool = False,          # off by default (not consistently useful)
        n_spectral_contrast_bands: int = 6,
        # Prosodic
        use_prosodic: bool = True,
        # Summary statistics per time-series feature
        stat_functions: Tuple[str, ...] = ("mean", "std", "median", "p25", "p75", "max"),
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.use_mfcc_delta = use_mfcc_delta
        self.use_mfcc_delta2 = use_mfcc_delta2

        self.use_lfcc = use_lfcc
        self.n_lfcc = n_lfcc
        self.use_lfcc_delta = use_lfcc_delta
        self.use_lfcc_delta2 = use_lfcc_delta2

        self.use_spectral = use_spectral
        self.n_chroma = n_chroma
        self.use_chroma = use_chroma
        self.n_spectral_contrast_bands = n_spectral_contrast_bands

        self.use_prosodic = use_prosodic
        self.stat_functions = stat_functions

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        c = cls()
        for k, v in d.items():
            setattr(c, k, v)
        return c


# ─────────────────────────────────────────────
# Summary statistics helpers
# ─────────────────────────────────────────────

def _stats(x: np.ndarray, stat_functions: Tuple[str, ...]) -> np.ndarray:
    """
    Compute a vector of summary statistics for a 1-D or 2-D array.

    If x is 2-D (n_features × n_frames), statistics are computed along
    the frames axis (axis=1) and then flattened.
    """
    if x.ndim == 1:
        x = x[np.newaxis, :]  # (1, n_frames)

    out = []
    for fn in stat_functions:
        if fn == "mean":
            out.append(np.mean(x, axis=1))
        elif fn == "std":
            out.append(np.std(x, axis=1))
        elif fn == "median":
            out.append(np.median(x, axis=1))
        elif fn == "p25":
            out.append(np.percentile(x, 25, axis=1))
        elif fn == "p75":
            out.append(np.percentile(x, 75, axis=1))
        elif fn == "max":
            out.append(np.max(x, axis=1))
        elif fn == "min":
            out.append(np.min(x, axis=1))
        elif fn == "range":
            out.append(np.ptp(x, axis=1))
        else:
            raise ValueError(f"Unknown stat function: {fn}")

    return np.concatenate(out).astype(np.float32)


def _safe(x: np.ndarray) -> np.ndarray:
    """Replace NaN and Inf with 0."""
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


# ─────────────────────────────────────────────
# A. MFCC
# ─────────────────────────────────────────────

def extract_mfcc(
    waveform: np.ndarray,
    cfg: FeatureConfig,
) -> np.ndarray:
    """
    Extract MFCC features.

    MFCCs capture the spectral envelope of speech using a mel-scale
    filterbank followed by a Discrete Cosine Transform (DCT). They are
    among the most discriminative features for distinguishing natural
    from synthesised speech because vocoders produce unnatural patterns
    in the mel spectrum.

    Returns a 1-D feature vector of summary statistics over time.
    """
    if not _LIBROSA_AVAILABLE:
        raise RuntimeError("librosa required for MFCC extraction.")

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=cfg.sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
    )  # shape: (n_mfcc, T)

    parts = [_stats(mfcc, cfg.stat_functions)]

    if cfg.use_mfcc_delta:
        delta = librosa.feature.delta(mfcc, order=1)
        parts.append(_stats(delta, cfg.stat_functions))

    if cfg.use_mfcc_delta2:
        delta2 = librosa.feature.delta(mfcc, order=2)
        parts.append(_stats(delta2, cfg.stat_functions))

    return _safe(np.concatenate(parts))


# ─────────────────────────────────────────────
# B. LFCC
# ─────────────────────────────────────────────

def _linear_filterbank(
    n_filters: int, n_fft: int, sr: int
) -> np.ndarray:
    """
    Construct a linear-frequency filterbank (n_filters × (n_fft//2 + 1)).

    Unlike the Mel filterbank, the filters are equally spaced on the
    linear frequency axis. LFCC features are known to capture low-level
    spectral artefacts of TTS/VC systems that MFCC misses because the
    mel scale compresses the high-frequency region.
    """
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freqs)
    filter_edges = np.linspace(0, sr / 2, n_filters + 2)

    fb = np.zeros((n_filters, n_freqs), dtype=np.float32)
    for m in range(n_filters):
        f_m_minus = filter_edges[m]
        f_m = filter_edges[m + 1]
        f_m_plus = filter_edges[m + 2]

        up = (freqs - f_m_minus) / (f_m - f_m_minus + 1e-12)
        down = (f_m_plus - freqs) / (f_m_plus - f_m + 1e-12)
        fb[m] = np.where(
            (freqs >= f_m_minus) & (freqs <= f_m), up,
            np.where((freqs >= f_m) & (freqs <= f_m_plus), down, 0.0)
        )
    return fb


def extract_lfcc(
    waveform: np.ndarray,
    cfg: FeatureConfig,
) -> np.ndarray:
    """
    Extract LFCC features using a linear-frequency filterbank.

    Implementation follows the same pipeline as MFCC:
    STFT → power spectrum → linear filterbank → log → DCT.

    Several ASVspoof papers (e.g., Sahidullah et al., 2015) show that LFCC
    outperforms MFCC for detecting TTS-based spoof attacks because the
    linear scale preserves high-frequency vocoder artefacts.

    Returns a 1-D feature vector of summary statistics over time.
    """
    if not _LIBROSA_AVAILABLE:
        raise RuntimeError("librosa required for LFCC extraction.")

    # 1. STFT power spectrum: shape (n_fft//2+1, T)
    S = np.abs(librosa.stft(
        waveform,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
    )) ** 2

    # 2. Apply linear filterbank
    n_filters = max(cfg.n_lfcc * 2, 40)
    fb = _linear_filterbank(n_filters, cfg.n_fft, cfg.sr)   # (n_filters, F)
    filtered = np.dot(fb, S)                                 # (n_filters, T)
    filtered = np.maximum(filtered, 1e-10)
    log_filtered = np.log(filtered)                          # log energy

    # 3. DCT to get cepstral coefficients
    lfcc = dct(log_filtered, type=2, axis=0, norm="ortho")[: cfg.n_lfcc]  # (n_lfcc, T)

    parts = [_stats(lfcc, cfg.stat_functions)]

    if cfg.use_lfcc_delta:
        delta = np.diff(lfcc, axis=1, prepend=lfcc[:, :1])
        parts.append(_stats(delta, cfg.stat_functions))

    if cfg.use_lfcc_delta2:
        delta2 = np.diff(lfcc, n=2, axis=1, prepend=lfcc[:, :2])
        parts.append(_stats(delta2, cfg.stat_functions))

    return _safe(np.concatenate(parts))


# ─────────────────────────────────────────────
# C. Spectral features
# ─────────────────────────────────────────────

def extract_spectral_features(
    waveform: np.ndarray,
    cfg: FeatureConfig,
) -> np.ndarray:
    """
    Extract spectral and temporal features:

    - Spectral centroid    : weighted mean frequency → timbre brightness
    - Spectral bandwidth   : weighted spread around centroid
    - Spectral rolloff     : frequency below which 85% of energy lies
    - Spectral flatness    : ratio of geometric to arithmetic mean → tonality
    - Spectral contrast    : difference between peaks and valleys per subband
    - Zero-crossing rate   : proxy for noisiness / frication

    Returns a 1-D feature vector of summary statistics over time.
    """
    if not _LIBROSA_AVAILABLE:
        raise RuntimeError("librosa required for spectral feature extraction.")

    parts = []

    # Centroid
    centroid = librosa.feature.spectral_centroid(
        y=waveform, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length
    )
    parts.append(_stats(centroid, cfg.stat_functions))

    # Bandwidth
    bw = librosa.feature.spectral_bandwidth(
        y=waveform, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length
    )
    parts.append(_stats(bw, cfg.stat_functions))

    # Rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=waveform, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        roll_percent=0.85,
    )
    parts.append(_stats(rolloff, cfg.stat_functions))

    # Flatness
    flatness = librosa.feature.spectral_flatness(
        y=waveform, n_fft=cfg.n_fft, hop_length=cfg.hop_length
    )
    parts.append(_stats(flatness, cfg.stat_functions))

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(
        y=waveform, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_bands=cfg.n_spectral_contrast_bands,
    )
    parts.append(_stats(contrast, cfg.stat_functions))

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        waveform, hop_length=cfg.hop_length
    )
    parts.append(_stats(zcr, cfg.stat_functions))

    # Optional chroma
    if cfg.use_chroma:
        chroma = librosa.feature.chroma_stft(
            y=waveform, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_chroma=cfg.n_chroma,
        )
        parts.append(_stats(chroma, cfg.stat_functions))

    return _safe(np.concatenate(parts))


# ─────────────────────────────────────────────
# D. Prosodic / temporal features
# ─────────────────────────────────────────────

def extract_prosodic_features(
    waveform: np.ndarray,
    cfg: FeatureConfig,
) -> np.ndarray:
    """
    Extract prosodic and temporal features:

    - RMS energy per frame + statistics
    - Pitch (fundamental frequency) estimates + statistics
    - Voiced/unvoiced ratio
    - Temporal variation of energy (how dynamic the energy is)

    Synthetic speech often has unnaturally flat pitch and energy profiles,
    making these features informative.

    Returns a 1-D feature vector.
    """
    if not _LIBROSA_AVAILABLE:
        raise RuntimeError("librosa required for prosodic feature extraction.")

    parts = []

    # RMS energy
    rms = librosa.feature.rms(
        y=waveform, frame_length=cfg.n_fft, hop_length=cfg.hop_length
    )[0]
    parts.append(_stats(rms[np.newaxis, :], cfg.stat_functions))

    # Temporal variation of energy
    energy_diff = np.diff(rms, prepend=rms[:1])
    parts.append(_stats(energy_diff[np.newaxis, :], cfg.stat_functions))

    # Pitch (pyin is more accurate but slow; yin is used here)
    try:
        f0, voiced_flag, _ = librosa.pyin(
            waveform,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=cfg.sr,
            hop_length=cfg.hop_length,
        )
        voiced_ratio = np.mean(voiced_flag.astype(float))
        # Replace NaN in f0 (unvoiced frames) with 0 for statistics
        f0_clean = np.where(voiced_flag, f0, 0.0)
        parts.append(_stats(f0_clean[np.newaxis, :], cfg.stat_functions))
        parts.append(np.array([voiced_ratio], dtype=np.float32))
    except Exception:
        # Pitch estimation can fail on very short or noisy clips
        n_pitch_stats = len(cfg.stat_functions)
        parts.append(np.zeros(n_pitch_stats, dtype=np.float32))
        parts.append(np.zeros(1, dtype=np.float32))

    return _safe(np.concatenate(parts))


# ─────────────────────────────────────────────
# Master extraction function
# ─────────────────────────────────────────────

def extract_all_features(
    waveform: np.ndarray,
    cfg: Optional[FeatureConfig] = None,
) -> np.ndarray:
    """
    Extract the full feature vector for a single waveform segment.

    Parameters
    ----------
    waveform : np.ndarray
        1-D float32 audio signal at cfg.sr Hz.
    cfg : FeatureConfig, optional
        Feature configuration. Defaults to FeatureConfig() if None.

    Returns
    -------
    features : np.ndarray
        Fixed-length 1-D float32 array.
    """
    if cfg is None:
        cfg = FeatureConfig()

    parts = []

    if cfg.use_mfcc:
        parts.append(extract_mfcc(waveform, cfg))

    if cfg.use_lfcc:
        parts.append(extract_lfcc(waveform, cfg))

    if cfg.use_spectral:
        parts.append(extract_spectral_features(waveform, cfg))

    if cfg.use_prosodic:
        parts.append(extract_prosodic_features(waveform, cfg))

    if not parts:
        raise ValueError("All feature groups are disabled. Enable at least one.")

    return _safe(np.concatenate(parts))


def compute_expected_feature_dim(cfg: Optional[FeatureConfig] = None) -> int:
    """
    Compute the expected feature vector dimensionality for a given config
    without loading any audio. Useful for sanity checks.
    """
    if cfg is None:
        cfg = FeatureConfig()

    n_stats = len(cfg.stat_functions)
    dim = 0

    if cfg.use_mfcc:
        multiplier = 1 + cfg.use_mfcc_delta + cfg.use_mfcc_delta2
        dim += cfg.n_mfcc * n_stats * multiplier

    if cfg.use_lfcc:
        multiplier = 1 + cfg.use_lfcc_delta + cfg.use_lfcc_delta2
        dim += cfg.n_lfcc * n_stats * multiplier

    if cfg.use_spectral:
        # centroid, bw, rolloff, flatness → 1 × n_stats each
        # contrast → (n_bands+1) × n_stats
        # zcr → 1 × n_stats
        # chroma → n_chroma × n_stats  (if enabled)
        dim += (4 + (cfg.n_spectral_contrast_bands + 1) + 1) * n_stats
        if cfg.use_chroma:
            dim += cfg.n_chroma * n_stats

    if cfg.use_prosodic:
        # rms, energy_diff → 1 × n_stats each
        # f0 → 1 × n_stats
        # voiced_ratio → 1
        dim += 3 * n_stats + 1

    return dim


# ─────────────────────────────────────────────
# Batch extraction
# ─────────────────────────────────────────────

def extract_features_batch(
    waveforms: List[np.ndarray],
    cfg: Optional[FeatureConfig] = None,
    n_segments_per_file: int = 1,
    verbose: bool = True,
) -> np.ndarray:
    """
    Extract features from a list of waveforms.

    Each waveform is treated as one sample; for multi-segment files,
    call this function on each segment and aggregate externally.

    Returns
    -------
    X : np.ndarray shape (n_samples, n_features)
    """
    if cfg is None:
        cfg = FeatureConfig()

    X = []
    for i, waveform in enumerate(waveforms):
        try:
            feat = extract_all_features(waveform, cfg)
            X.append(feat)
        except Exception as exc:
            logger.warning(f"Feature extraction failed for sample {i}: {exc}")
            # Append a zero vector so indices stay aligned
            dim = compute_expected_feature_dim(cfg)
            X.append(np.zeros(dim, dtype=np.float32))

        if verbose and (i + 1) % 500 == 0:
            logger.info(f"Features extracted: {i+1}/{len(waveforms)}")

    return np.array(X, dtype=np.float32)


def aggregate_segment_features(
    segment_features: np.ndarray,
) -> np.ndarray:
    """
    Aggregate features across multiple segments of the same recording.

    Strategy: concatenate [mean, std] across segments to capture
    both typical behaviour and within-recording variation.

    Parameters
    ----------
    segment_features : np.ndarray shape (n_segments, n_features)

    Returns
    -------
    aggregated : np.ndarray shape (2 * n_features,)
    """
    if segment_features.ndim == 1:
        return segment_features

    mean_feat = np.mean(segment_features, axis=0)
    std_feat = np.std(segment_features, axis=0)
    return _safe(np.concatenate([mean_feat, std_feat]))
