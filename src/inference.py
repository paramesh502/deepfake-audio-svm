"""
inference.py
============
Standalone inference pipeline for audio deepfake detection.

The function predict_audio() accepts any audio file path and returns
a structured result dict. It does NOT require the file to be in any
training dataset — it processes arbitrary user-provided audio.

The pipeline reproduces training-time preprocessing exactly by using
the same FeatureConfig and scaler that were saved during training.
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

from .preprocessing import (
    preprocess_audio_file,
    TARGET_SR,
    SEGMENT_DURATION,
    SEGMENT_HOP,
)
from .features import (
    extract_all_features,
    aggregate_segment_features,
    FeatureConfig,
)

logger = logging.getLogger(__name__)

# Confidence thresholds for uncertainty reporting
HIGH_CONF_THRESHOLD = 0.75   # p_fake or p_real > 0.75 → confident
LOW_CONF_THRESHOLD = 0.60    # between 0.60 and 0.75 → moderate confidence
# Below LOW_CONF_THRESHOLD → uncertain prediction


class InferencePipeline:
    """
    Encapsulates the full inference pipeline after training.

    Load from a saved joblib bundle using InferencePipeline.load().

    Attributes
    ----------
    model      : fitted classifier with predict_proba method
    scaler     : fitted StandardScaler (may be inside the model pipeline)
    imputer    : fitted SimpleImputer (may be inside the model pipeline)
    cfg        : FeatureConfig used at training time
    model_name : string identifier
    metadata   : dict of model metadata (version, dataset, training date, etc.)
    calibrator : PlattCalibrator or None
    """

    def __init__(
        self,
        model,
        cfg: FeatureConfig,
        model_name: str = "unknown",
        metadata: Optional[Dict] = None,
        scaler=None,
        imputer=None,
        calibrator=None,
    ):
        self.model = model
        self.cfg = cfg
        self.model_name = model_name
        self.metadata = metadata or {}
        self.scaler = scaler
        self.imputer = imputer
        self.calibrator = calibrator

    def _get_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get calibrated probabilities.

        Tries in order:
        1. self.calibrator (PlattCalibrator for scratch SVM)
        2. self.model.predict_proba (sklearn pipelines with probability=True)
        3. Sigmoid of decision_function (fallback for LinearSVC)

        Returns array shape (n, 2): [P(REAL), P(FAKE)]
        """
        if self.imputer is not None:
            X = self.imputer.transform(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)

        if self.calibrator is not None:
            # Scratch SVM path: get decision scores then apply Platt scaling
            scores = self.model.decision_function(X)
            return self.calibrator.predict_proba(scores)

        try:
            proba = self.model.predict_proba(X)
            # Verify column order: sklearn returns proba in class-label order
            # For binary {0,1}: column 0 = P(REAL), column 1 = P(FAKE)
            return proba.astype(np.float32)
        except AttributeError:
            pass

        # Fallback: use sigmoid of decision function
        try:
            scores = self.model.decision_function(X)
            if scores.ndim == 1:
                p_fake = 1.0 / (1.0 + np.exp(-scores))
                return np.column_stack([1 - p_fake, p_fake]).astype(np.float32)
        except AttributeError:
            pass

        raise RuntimeError(
            "Cannot compute probabilities: model has no predict_proba, "
            "calibrator, or decision_function."
        )

    def predict_single(
        self,
        waveform: np.ndarray,
        sr: int = TARGET_SR,
        segment: bool = True,
    ) -> Dict[str, Any]:
        """
        Predict on a single (preprocessed) waveform.

        Returns
        -------
        dict with keys:
            label_int      : 0 (REAL) or 1 (FAKE)
            label_str      : "REAL" or "FAKE"
            p_real         : float in [0, 1]
            p_fake         : float in [0, 1]
            confidence     : "high" | "moderate" | "uncertain"
            warning        : str (uncertainty message if applicable)
            segment_proba  : list of (p_real, p_fake) per segment
        """
        from .preprocessing import segment_waveform

        if segment:
            segs = segment_waveform(waveform, sr, SEGMENT_DURATION, SEGMENT_HOP)
        else:
            segs = [waveform]

        # Extract features per segment
        seg_features = []
        for seg in segs:
            try:
                feat = extract_all_features(seg, self.cfg)
                seg_features.append(feat)
            except Exception as exc:
                logger.warning(f"Feature extraction failed for segment: {exc}")

        if not seg_features:
            return self._error_result("Feature extraction failed for all segments.")

        X_segs = np.array(seg_features, dtype=np.float32)

        # Get per-segment probabilities
        try:
            seg_proba = self._get_proba(X_segs)   # (n_segs, 2)
        except Exception as exc:
            return self._error_result(f"Probability estimation failed: {exc}")

        segment_proba = [(float(p[0]), float(p[1])) for p in seg_proba]

        # Aggregate: mean probability across segments (soft voting)
        p_real = float(np.mean(seg_proba[:, 0]))
        p_fake = float(np.mean(seg_proba[:, 1]))

        # Renormalise to sum to 1 (guards against floating-point drift)
        total = p_real + p_fake
        if total > 0:
            p_real /= total
            p_fake /= total

        label_int = 1 if p_fake >= 0.5 else 0
        label_str = "FAKE" if label_int == 1 else "REAL"

        max_p = max(p_real, p_fake)
        if max_p >= HIGH_CONF_THRESHOLD:
            confidence = "high"
            warning = ""
        elif max_p >= LOW_CONF_THRESHOLD:
            confidence = "moderate"
            warning = (
                "Moderate confidence — the model found weak evidence. "
                "Additional verification is recommended."
            )
        else:
            confidence = "uncertain"
            warning = (
                "Uncertain prediction — the audio characteristics are ambiguous. "
                "The model cannot reliably classify this sample. "
                "Human review is recommended."
            )

        return {
            "label_int": label_int,
            "label_str": label_str,
            "p_real": p_real,
            "p_fake": p_fake,
            "confidence": confidence,
            "warning": warning,
            "segment_proba": segment_proba,
            "n_segments": len(segs),
            "error": None,
        }

    @staticmethod
    def _error_result(msg: str) -> Dict[str, Any]:
        return {
            "label_int": None,
            "label_str": None,
            "p_real": None,
            "p_fake": None,
            "confidence": None,
            "warning": "",
            "segment_proba": [],
            "n_segments": 0,
            "error": msg,
        }

    # ─────────────────────────────────
    # Save / Load
    # ─────────────────────────────────

    def save(self, path: str) -> None:
        """Save all inference artifacts to a joblib bundle."""
        import joblib
        bundle = {
            "model": self.model,
            "cfg": self.cfg.to_dict(),
            "model_name": self.model_name,
            "metadata": self.metadata,
            "scaler": self.scaler,
            "imputer": self.imputer,
            "calibrator": (
                self.calibrator.to_dict() if self.calibrator is not None else None
            ),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(bundle, path, compress=3)
        logger.info(f"Inference pipeline saved to {path}")

    @classmethod
    def load(cls, path: str) -> "InferencePipeline":
        """Load an inference pipeline from a joblib bundle."""
        import joblib
        from .calibration import PlattCalibrator

        bundle = joblib.load(path)
        cfg = FeatureConfig.from_dict(bundle["cfg"])

        calibrator = None
        if bundle.get("calibrator") is not None:
            calibrator = PlattCalibrator.from_dict(bundle["calibrator"])

        pipeline = cls(
            model=bundle["model"],
            cfg=cfg,
            model_name=bundle.get("model_name", "unknown"),
            metadata=bundle.get("metadata", {}),
            scaler=bundle.get("scaler"),
            imputer=bundle.get("imputer"),
            calibrator=calibrator,
        )
        logger.info(f"Inference pipeline loaded from {path}")
        return pipeline


# ─────────────────────────────────────────────
# High-level function: predict_audio
# ─────────────────────────────────────────────

def predict_audio(
    file_path: str,
    pipeline: InferencePipeline,
    target_sr: int = TARGET_SR,
) -> Dict[str, Any]:
    """
    Main inference entry point.

    Accepts any audio file (WAV, MP3, FLAC, OGG, M4A) and returns
    a structured result dict.

    Parameters
    ----------
    file_path : str
        Path to the audio file.  Does NOT need to be from any dataset.
    pipeline  : InferencePipeline
        Loaded inference pipeline.

    Returns
    -------
    dict with:
        file_path      : str
        valid          : bool
        label_str      : "REAL" | "FAKE" | None
        label_int      : 0 | 1 | None
        p_real         : float | None
        p_fake         : float | None
        confidence     : str | None
        warning        : str
        diagnostics    : dict (duration, clipping, speech_activity, ...)
        segment_proba  : list of (p_real, p_fake)
        error          : str | None
    """
    # 1. Preprocessing
    prep = preprocess_audio_file(
        file_path,
        target_sr=target_sr,
        normalize=True,
        segment=True,
    )

    if not prep["valid"]:
        return {
            "file_path": file_path,
            "valid": False,
            "label_str": None,
            "label_int": None,
            "p_real": None,
            "p_fake": None,
            "confidence": None,
            "warning": "; ".join(prep["errors"]),
            "diagnostics": prep.get("diagnostics", {}),
            "segment_proba": [],
            "error": "; ".join(prep["errors"]),
        }

    warnings_list = prep["warnings"]

    # 2. Inference
    pred = pipeline.predict_single(
        waveform=prep["waveform"],
        sr=prep["sr"],
        segment=True,
    )

    if pred["error"] is not None:
        return {
            "file_path": file_path,
            "valid": False,
            "label_str": None,
            "label_int": None,
            "p_real": None,
            "p_fake": None,
            "confidence": None,
            "warning": pred["error"],
            "diagnostics": prep.get("diagnostics", {}),
            "segment_proba": [],
            "error": pred["error"],
        }

    # Merge preprocessing warnings with prediction warning
    all_warnings = warnings_list + ([pred["warning"]] if pred["warning"] else [])

    return {
        "file_path": file_path,
        "valid": True,
        "label_str": pred["label_str"],
        "label_int": pred["label_int"],
        "p_real": pred["p_real"],
        "p_fake": pred["p_fake"],
        "confidence": pred["confidence"],
        "warning": " | ".join(all_warnings) if all_warnings else "",
        "diagnostics": {
            **prep.get("diagnostics", {}),
            "duration_s": prep["duration_s"],
            "sr": prep["sr"],
            "n_segments": pred["n_segments"],
        },
        "segment_proba": pred["segment_proba"],
        "error": None,
    }


# ─────────────────────────────────────────────
# Non-speech / OOD detection
# ─────────────────────────────────────────────

def check_is_speech(
    waveform: np.ndarray,
    sr: int,
    min_speech_ratio: float = 0.10,
) -> Dict[str, Any]:
    """
    Quick check for whether the input contains sufficient speech.

    This is a heuristic based on energy and zero-crossing rate, not a
    guaranteed speech detector.  It is presented to the user as a
    quality warning, not a mathematically guaranteed OOD detector.
    """
    from .preprocessing import check_silence
    result = check_silence(waveform, sr)
    has_speech = result["speech_ratio"] >= min_speech_ratio
    return {
        "has_speech": has_speech,
        "speech_ratio": result["speech_ratio"],
        "warning": (
            "Input does not appear to contain sufficient speech. "
            "Classification may be unreliable."
            if not has_speech else ""
        ),
    }
