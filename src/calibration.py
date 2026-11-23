"""
calibration.py
==============
Probability calibration for SVM decision scores.

Why calibration?
----------------
Raw SVM decision scores  f(x) = w^T x + b  are unbounded real numbers.
They convey ranking information but their magnitude does not correspond
to class probabilities. Platt (1999) proposed fitting a sigmoid function:

    P(y=1 | f) = 1 / (1 + exp(A·f + B))

on a held-out calibration set, where A and B are fitted by maximum
likelihood. This converts scores into well-calibrated probabilities.

Scikit-learn wraps this in CalibratedClassifierCV. For the scratch SVM,
we fit A and B directly using scipy.optimize.minimize.

Evaluation
----------
- Brier score: mean squared error between probabilities and true labels.
  Lower is better. Perfect calibration → 0.
- Log loss: negative log-likelihood. Lower is better.
- Reliability diagram: compare predicted probabilities against empirical
  fraction of positives in equal-width bins.
- Expected Calibration Error (ECE): weighted mean absolute deviation
  between mean predicted probability and observed fraction per bin.
"""

import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Platt scaling for scratch SVM
# ─────────────────────────────────────────────

class PlattCalibrator:
    """
    Fits Platt scaling parameters (A, B) for a pre-trained binary classifier.

    Usage
    -----
    cal = PlattCalibrator()
    cal.fit(scores_cal, y_cal)
    proba = cal.predict_proba(scores_new)
    """

    def __init__(self):
        self.A_: Optional[float] = None
        self.B_: Optional[float] = None
        self.fitted_: bool = False

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        """
        Fit Platt scaling parameters on calibration data.

        Parameters
        ----------
        scores : np.ndarray, shape (n,)
            Decision function values from the classifier.
        y : np.ndarray, shape (n,)
            True binary labels in {0, 1}.
        """
        y = np.asarray(y, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)

        # Label smoothing to prevent over-confidence (Platt's original trick)
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        t_pos = (n_pos + 1) / (n_pos + 2)
        t_neg = 1.0 / (n_neg + 2)
        t = np.where(y == 1, t_pos, t_neg)

        def neg_log_likelihood(params):
            A, B = params
            p = expit(-(A * scores + B))
            eps = 1e-12
            nll = -np.sum(t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps))
            return nll

        result = minimize(
            neg_log_likelihood,
            x0=[0.0, np.log((n_neg + 1) / (n_pos + 1))],
            method="L-BFGS-B",
        )

        self.A_, self.B_ = result.x
        self.fitted_ = True
        logger.info(f"Platt calibration: A={self.A_:.4f}  B={self.B_:.4f}")
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """
        Return calibrated probabilities.

        Returns
        -------
        np.ndarray, shape (n, 2)
            Columns: [P(y=0), P(y=1)]
        """
        if not self.fitted_:
            raise RuntimeError("PlattCalibrator not fitted yet.")
        scores = np.asarray(scores, dtype=np.float64)
        p_fake = expit(-(self.A_ * scores + self.B_))
        return np.column_stack([1 - p_fake, p_fake]).astype(np.float32)

    def predict_proba_pair(self, score: float) -> Tuple[float, float]:
        """Return (p_real, p_fake) for a single score."""
        proba = self.predict_proba(np.array([score]))
        return float(proba[0, 0]), float(proba[0, 1])

    def to_dict(self) -> dict:
        return {"A": self.A_, "B": self.B_}

    @classmethod
    def from_dict(cls, d: dict) -> "PlattCalibrator":
        cal = cls()
        cal.A_ = d["A"]
        cal.B_ = d["B"]
        cal.fitted_ = True
        return cal


# ─────────────────────────────────────────────
# Calibration evaluation metrics
# ─────────────────────────────────────────────

def compute_brier_score(
    y_true: np.ndarray,
    proba: np.ndarray,
) -> float:
    """
    Brier score = (1/n) Σ (p_predicted - y_true)^2

    Ranges from 0 (perfect) to 1 (worst).
    A perfectly calibrated naive predictor that always predicts the
    base rate p achieves Brier score = p(1-p).
    """
    y_true = np.asarray(y_true)
    p_pos = np.asarray(proba[:, 1] if proba.ndim == 2 else proba)
    return float(brier_score_loss(y_true, p_pos))


def compute_log_loss(
    y_true: np.ndarray,
    proba: np.ndarray,
) -> float:
    """
    Log loss (cross-entropy) = -(1/n) Σ [y log(p) + (1-y) log(1-p)]

    Lower is better. Perfectly calibrated predictions achieve lower log loss.
    """
    y_true = np.asarray(y_true)
    return float(log_loss(y_true, proba))


def compute_ece(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    ECE = Σ_b (|B_b| / n) · |acc(B_b) − conf(B_b)|

    where:
      B_b   = set of samples whose predicted probability falls in bin b
      acc   = fraction of positive samples in the bin
      conf  = mean predicted probability in the bin
      n     = total number of samples

    A perfectly calibrated model has ECE = 0.
    ECE measures the average gap between confidence and accuracy.
    """
    y_true = np.asarray(y_true)
    p_pos = np.asarray(proba[:, 1] if proba.ndim == 2 else proba)
    n = len(y_true)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p_pos >= lo) & (p_pos < hi)
        if i == n_bins - 1:
            mask = (p_pos >= lo) & (p_pos <= hi)
        n_b = mask.sum()
        if n_b == 0:
            continue
        acc_b = y_true[mask].mean()
        conf_b = p_pos[mask].mean()
        ece += (n_b / n) * abs(acc_b - conf_b)
    return float(ece)


def reliability_diagram_data(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute data for a reliability (calibration) diagram.

    Returns
    -------
    mean_predicted_prob : np.ndarray  (x-axis)
    fraction_of_positives : np.ndarray  (y-axis)
    bin_counts : np.ndarray
    """
    y_true = np.asarray(y_true)
    p_pos = np.asarray(proba[:, 1] if proba.ndim == 2 else proba)
    fraction_of_positives, mean_predicted_prob = calibration_curve(
        y_true, p_pos, n_bins=n_bins, strategy="uniform"
    )
    # Bin counts
    bins = np.linspace(0, 1, n_bins + 1)
    bin_counts = np.histogram(p_pos, bins=bins)[0]
    return mean_predicted_prob, fraction_of_positives, bin_counts


def evaluate_calibration(
    y_true: np.ndarray,
    proba_uncal: np.ndarray,
    proba_cal: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compare calibration metrics before and after Platt scaling.

    Returns a summary dict for reporting.
    """
    return {
        "before": {
            "brier_score": compute_brier_score(y_true, proba_uncal),
            "log_loss": compute_log_loss(y_true, proba_uncal),
            "ece": compute_ece(y_true, proba_uncal, n_bins),
        },
        "after": {
            "brier_score": compute_brier_score(y_true, proba_cal),
            "log_loss": compute_log_loss(y_true, proba_cal),
            "ece": compute_ece(y_true, proba_cal, n_bins),
        },
    }


# ─────────────────────────────────────────────
# Calibrate a scikit-learn pipeline
# ─────────────────────────────────────────────

def calibrate_sklearn_pipeline(
    pipeline,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "sigmoid",
    cv: str = "prefit",
) -> CalibratedClassifierCV:
    """
    Wrap a fitted sklearn pipeline with Platt scaling.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    X_cal    : calibration feature matrix
    y_cal    : calibration labels
    method   : "sigmoid" (Platt) or "isotonic"
    cv       : "prefit" uses the already-fitted estimator.

    Returns
    -------
    CalibratedClassifierCV  (fitted)
    """
    cal_clf = CalibratedClassifierCV(pipeline, method=method, cv=cv)
    cal_clf.fit(X_cal, y_cal)
    return cal_clf
