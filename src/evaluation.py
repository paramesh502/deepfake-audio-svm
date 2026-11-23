"""
evaluation.py
=============
Comprehensive evaluation metrics for binary deepfake speech detection.

Metrics computed
----------------
- Accuracy, Balanced Accuracy
- Precision, Recall, F1 (positive class = FAKE = 1)
- Specificity (True Negative Rate)
- Confusion Matrix
- ROC Curve and AUC
- Precision-Recall Curve and Average Precision
- Equal Error Rate (EER)
- False Positive Rate, False Negative Rate
- Brier Score, Log Loss
- Confidence Intervals via bootstrap (optional)

Label conventions
-----------------
0 = REAL
1 = FAKE
"""

import logging
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    log_loss,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    zero_division: int = 0,
) -> Dict[str, Any]:
    """
    Compute all classification metrics.

    Parameters
    ----------
    y_true   : true binary labels {0, 1}
    y_pred   : predicted binary labels {0, 1}
    y_scores : predicted probability for the positive class (FAKE).
               Required for ROC-AUC, PR-AUC, EER, Brier, LogLoss.

    Returns
    -------
    dict with all metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    n = len(y_true)
    accuracy      = accuracy_score(y_true, y_pred)
    bal_accuracy  = balanced_accuracy_score(y_true, y_pred)
    precision     = precision_score(y_true, y_pred, zero_division=zero_division)
    recall        = recall_score(y_true, y_pred, zero_division=zero_division)
    f1            = f1_score(y_true, y_pred, zero_division=zero_division)
    specificity   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr_val       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_val       = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    result = {
        "n_samples": n,
        "n_real": int(np.sum(y_true == 0)),
        "n_fake": int(np.sum(y_true == 1)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(bal_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "fpr": float(fpr_val),
        "fnr": float(fnr_val),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": cm.tolist(),
    }

    if y_scores is not None:
        y_scores = np.asarray(y_scores, dtype=np.float64)

        # ROC
        try:
            fpr_arr, tpr_arr, _ = roc_curve(y_true, y_scores)
            auc = roc_auc_score(y_true, y_scores)
            result["roc_auc"] = float(auc)
            result["roc_fpr"] = fpr_arr.tolist()
            result["roc_tpr"] = tpr_arr.tolist()
        except Exception:
            result["roc_auc"] = None

        # Precision-Recall
        try:
            prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            result["average_precision"] = float(ap)
            result["pr_precision"] = prec_arr.tolist()
            result["pr_recall"] = rec_arr.tolist()
        except Exception:
            result["average_precision"] = None

        # Equal Error Rate
        result["eer"] = float(compute_eer(y_true, y_scores))

        # Brier score
        result["brier_score"] = float(brier_score_loss(y_true, y_scores))

        # Log loss
        try:
            # Avoid log(0) issues
            eps = 1e-7
            y_scores_clipped = np.clip(y_scores, eps, 1 - eps)
            result["log_loss"] = float(log_loss(y_true, y_scores_clipped))
        except Exception:
            result["log_loss"] = None

    return result


# ─────────────────────────────────────────────
# Equal Error Rate
# ─────────────────────────────────────────────

def compute_eer(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """
    Compute the Equal Error Rate (EER).

    EER is the operating point where the False Acceptance Rate (FAR)
    equals the False Rejection Rate (FRR). In binary detection:
        FAR = FPR = FP / (FP + TN)   (fake accepted as real)
        FRR = FNR = FN / (FN + TP)   (real rejected as fake)

    EER is a standard metric in speaker verification and anti-spoofing.
    Lower EER → better discrimination.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    # Find the threshold where FPR ≈ FNR
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer)


# ─────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────

def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap confidence interval for a scalar metric.

    Parameters
    ----------
    metric_fn : callable(y_true, y_pred) → float

    Returns
    -------
    (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            score = metric_fn(y_true[idx], y_pred[idx])
            bootstrap_scores.append(score)
        except Exception:
            pass

    alpha = (1 - ci) / 2
    lower = float(np.percentile(bootstrap_scores, 100 * alpha))
    upper = float(np.percentile(bootstrap_scores, 100 * (1 - alpha)))
    return float(point), lower, upper


# ─────────────────────────────────────────────
# Multi-model comparison table
# ─────────────────────────────────────────────

def build_comparison_table(
    results_dict: Dict[str, Dict],
    metric_keys: List[str] = None,
) -> Dict[str, Any]:
    """
    Build a comparison table from a dict of {model_name: metrics_dict}.

    Returns a dict suitable for DataFrame construction:
    {metric: {model_name: value}, ...}
    """
    if metric_keys is None:
        metric_keys = [
            "accuracy", "balanced_accuracy", "precision",
            "recall", "f1", "specificity", "roc_auc", "eer",
            "brier_score", "average_precision",
        ]

    table = {m: {} for m in metric_keys}
    for model_name, metrics in results_dict.items():
        for m in metric_keys:
            table[m][model_name] = metrics.get(m, None)

    return table


# ─────────────────────────────────────────────
# Generalization gap analysis
# ─────────────────────────────────────────────

def generalization_gap(
    internal_metrics: Dict[str, Any],
    external_metrics: Dict[str, Any],
    keys: List[str] = None,
) -> Dict[str, float]:
    """
    Compute the gap between internal (held-out test) and external metrics.

    A large gap indicates domain mismatch or overfitting to dataset-specific
    properties (codec, speaker characteristics, recording conditions).
    """
    if keys is None:
        keys = ["accuracy", "f1", "roc_auc", "eer"]

    gaps = {}
    for k in keys:
        int_val = internal_metrics.get(k)
        ext_val = external_metrics.get(k)
        if int_val is not None and ext_val is not None:
            # For EER: lower is better, so gap is ext - int (positive means degradation)
            # For others: higher is better, so gap is int - ext (positive means degradation)
            if k == "eer":
                gaps[k] = float(ext_val - int_val)
            else:
                gaps[k] = float(int_val - ext_val)
    return gaps
