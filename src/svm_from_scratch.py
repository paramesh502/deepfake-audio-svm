"""
svm_from_scratch.py
===================
Linear Soft-Margin Support Vector Machine implemented from scratch
using NumPy and mini-batch subgradient descent.

Mathematical Background
-----------------------
Given training data  {(x_i, y_i)}_{i=1}^{n}  with  y_i ∈ {-1, +1},
the SVM seeks a hyperplane  f(x) = w^T x + b  that maximises the margin
2/||w||  subject to correct classification with slack.

Primal formulation (soft-margin):
    min_{w, b}  (1/2)||w||^2  +  C/n · Σ_i max(0, 1 − y_i(w^T x_i + b))

Equivalent slack-variable formulation:
    min_{w, b, ξ}  (1/2)||w||^2  +  C · Σ_i ξ_i
    subject to:    y_i(w^T x_i + b) ≥ 1 − ξ_i,   ξ_i ≥ 0

The hinge loss  L_i = max(0, 1 − y_i f(x_i))  is convex but
non-differentiable at y_i f(x_i) = 1.

Subgradient of the objective w.r.t. w:
    if  y_i(w^T x_i + b) ≥ 1  (correctly classified with margin):
        ∂L/∂w  =  w               (regularisation only)
    else:
        ∂L/∂w  =  w  −  (C/n) · y_i x_i   (regularisation + hinge)

Subgradient w.r.t. b:
    if  y_i(w^T x_i + b) ≥ 1 :  0
    else:                        −(C/n) · y_i

Learning-rate schedule (Pegasos-style):
    η_t = 1 / (λ · t)    where  λ = 1/C

The class uses {-1, +1} labels internally.
Input labels {0, 1} are converted automatically.
"""

import time
import logging
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class LinearSVMScratch:
    """
    Linear soft-margin SVM via mini-batch subgradient descent.

    Parameters
    ----------
    C : float
        Regularisation parameter. Smaller C → wider margin, more misclassifications.
    max_epochs : int
        Number of full passes through the training set.
    batch_size : int
        Mini-batch size. 1 = stochastic gradient descent.
    lr_schedule : str
        "pegasos"  : η_t = 1/(λ·t)  — theoretically motivated schedule.
        "constant" : constant learning rate lr_init.
        "decay"    : η_t = lr_init / sqrt(t)
    lr_init : float
        Initial learning rate (used for "constant" and "decay" schedules).
    tol : float
        Convergence threshold on relative change in objective value.
    random_state : int
        Seed for reproducibility.
    verbose : bool
        Print progress every few epochs.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_epochs: int = 100,
        batch_size: int = 32,
        lr_schedule: str = "pegasos",
        lr_init: float = 0.01,
        tol: float = 1e-5,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.C = C
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.lr_init = lr_init
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        # Fit outputs
        self.w_: Optional[np.ndarray] = None   # weight vector
        self.b_: float = 0.0                   # bias
        self.objective_history_: List[float] = []
        self.train_acc_history_: List[float] = []
        self.n_support_vectors_: int = 0
        self.fitted_: bool = False
        self.n_features_: int = 0
        self.classes_: np.ndarray = np.array([-1, 1])
        self.training_time_: float = 0.0

    # ─────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────

    @staticmethod
    def _convert_labels(y: np.ndarray) -> np.ndarray:
        """Convert {0, 1} labels to {-1, +1}."""
        y = np.asarray(y, dtype=np.float64)
        if set(np.unique(y)).issubset({0, 1}):
            y = 2 * y - 1   # 0 → -1, 1 → +1
        assert set(np.unique(y)).issubset({-1.0, 1.0}), \
            f"Labels must be in {{0,1}} or {{-1,+1}}, got {np.unique(y)}"
        return y

    def _compute_objective(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Primal objective:  (1/2)||w||^2  +  (C/n) Σ max(0, 1 − y_i f(x_i))
        """
        n = len(y)
        scores = X @ self.w_ + self.b_
        hinge = np.maximum(0.0, 1.0 - y * scores)
        return 0.5 * np.dot(self.w_, self.w_) + (self.C / n) * hinge.sum()

    def _get_lr(self, t: int) -> float:
        """Return the learning rate at global step t (1-indexed)."""
        lam = 1.0 / self.C
        if self.lr_schedule == "pegasos":
            return 1.0 / (lam * t)
        elif self.lr_schedule == "decay":
            return self.lr_init / np.sqrt(t)
        else:  # constant
            return self.lr_init

    # ─────────────────────────────────
    # Fit
    # ─────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVMScratch":
        """
        Train the SVM on (X, y).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
            Labels in {0,1} or {-1,+1}.
        """
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y_pm = self._convert_labels(y)
        n, d = X.shape

        self.n_features_ = d
        self.w_ = np.zeros(d, dtype=np.float64)
        self.b_ = 0.0
        self.objective_history_ = []
        self.train_acc_history_ = []

        t = 1   # global step counter (1-indexed for Pegasos schedule)
        prev_obj = np.inf
        t0 = time.perf_counter()

        for epoch in range(1, self.max_epochs + 1):
            idx = rng.permutation(n)

            for start in range(0, n, self.batch_size):
                batch_idx = idx[start: start + self.batch_size]
                X_b = X[batch_idx]
                y_b = y_pm[batch_idx]
                n_b = len(y_b)

                # Decision values for the batch
                scores = X_b @ self.w_ + self.b_

                # Identify margin-violating samples: y_i * f(x_i) < 1
                # (the "active" hinge terms)
                margin = y_b * scores
                violating = margin < 1.0

                # Subgradient of w:
                #   dw = w  −  (C/n_b) * Σ_{i:violating} y_i x_i
                dw = self.w_.copy()
                if violating.any():
                    dw -= (self.C / n_b) * (y_b[violating, np.newaxis] * X_b[violating]).sum(axis=0)

                # Subgradient of b:
                #   db = −(C/n_b) * Σ_{i:violating} y_i
                db = 0.0
                if violating.any():
                    db = -(self.C / n_b) * y_b[violating].sum()

                lr = self._get_lr(t)
                self.w_ -= lr * dw
                self.b_ -= lr * db
                t += 1

            # Record objective and accuracy every epoch
            obj = self._compute_objective(X, y_pm)
            self.objective_history_.append(obj)

            preds = np.sign(X @ self.w_ + self.b_)
            acc = np.mean(preds == y_pm)
            self.train_acc_history_.append(float(acc))

            if self.verbose and (epoch % max(1, self.max_epochs // 10) == 0):
                logger.info(
                    f"Epoch {epoch:4d}/{self.max_epochs}  "
                    f"obj={obj:.6f}  acc={acc:.4f}  lr={self._get_lr(t):.6f}"
                )

            # Convergence check
            if abs(prev_obj - obj) / (abs(prev_obj) + 1e-12) < self.tol:
                if self.verbose:
                    logger.info(f"Converged at epoch {epoch}.")
                break
            prev_obj = obj

        self.training_time_ = time.perf_counter() - t0

        # Identify approximate support vectors
        scores_all = X @ self.w_ + self.b_
        margin_all = y_pm * scores_all
        self.n_support_vectors_ = int(np.sum(margin_all <= 1.0 + 1e-4))

        self.fitted_ = True
        logger.info(
            f"Training complete in {self.training_time_:.2f}s. "
            f"Support vectors ≈ {self.n_support_vectors_}."
        )
        return self

    # ─────────────────────────────────
    # Predict
    # ─────────────────────────────────

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw decision scores w^T x + b."""
        self._check_fitted()
        return np.asarray(X, dtype=np.float64) @ self.w_ + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        Returns {0, 1} to be consistent with sklearn conventions.
        """
        scores = self.decision_function(X)
        y_pm = np.sign(scores)
        y_pm[y_pm == 0] = 1   # tie → positive class
        # Map back to {0, 1}
        return ((y_pm + 1) / 2).astype(int)

    def predict_proba_platt(
        self,
        X: np.ndarray,
        A: float,
        B: float,
    ) -> np.ndarray:
        """
        Platt scaling: sigmoid(A * f(x) + B).

        Parameters A and B are fitted by calibration.py using a
        separate calibration set.

        Returns array shape (n, 2) with columns [P(y=0), P(y=1)].
        """
        scores = self.decision_function(X)
        prob_fake = 1.0 / (1.0 + np.exp(A * scores + B))
        return np.column_stack([1 - prob_fake, prob_fake]).astype(np.float32)

    # ─────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────

    def _check_fitted(self):
        if not self.fitted_:
            raise RuntimeError("SVM is not fitted yet. Call fit() first.")

    def get_convergence_data(self) -> Dict[str, Any]:
        """Return training curves for plotting."""
        self._check_fitted()
        return {
            "objective": self.objective_history_,
            "train_acc": self.train_acc_history_,
            "n_support_vectors": self.n_support_vectors_,
            "training_time_s": self.training_time_,
        }

    def compute_margin_violations(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyse margin violations on a dataset.

        Returns counts of:
        - correctly classified inside margin (support-vector zone)
        - correctly classified outside margin
        - margin violations (hinge loss > 0)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        y_pm = self._convert_labels(y)
        scores = X @ self.w_ + self.b_
        margin_values = y_pm * scores

        return {
            "n_outside_margin": int(np.sum(margin_values > 1.0)),
            "n_on_margin": int(np.sum(np.abs(margin_values - 1.0) <= 1e-4)),
            "n_violations": int(np.sum(margin_values < 1.0)),
            "mean_hinge_loss": float(np.mean(np.maximum(0, 1 - margin_values))),
            "functional_margin": float(scores.min()),
        }

    def geometric_margin(self) -> float:
        """Return the geometric margin 1/||w||."""
        self._check_fitted()
        return 1.0 / (np.linalg.norm(self.w_) + 1e-12)

    def __repr__(self) -> str:
        return (
            f"LinearSVMScratch(C={self.C}, max_epochs={self.max_epochs}, "
            f"batch_size={self.batch_size}, lr_schedule='{self.lr_schedule}')"
        )


# ─────────────────────────────────────────────
# Gradient check utility
# ─────────────────────────────────────────────

def numerical_gradient_check(
    svm: LinearSVMScratch,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-5,
    n_check: int = 10,
    random_state: int = 0,
) -> Dict[str, float]:
    """
    Finite-difference gradient check for the SVM objective w.r.t. w.

    For each of n_check randomly selected weight dimensions, compute:
        grad_analytic = ∂obj/∂w_j   (from subgradient formula)
        grad_numeric  = (obj(w_j+ε) − obj(w_j−ε)) / (2ε)

    Returns max and mean relative errors.
    """
    rng = np.random.default_rng(random_state)
    svm_copy = LinearSVMScratch(C=svm.C, max_epochs=1, verbose=False)
    svm_copy.w_ = svm.w_.copy()
    svm_copy.b_ = svm.b_
    svm_copy.fitted_ = True

    X = np.asarray(X, dtype=np.float64)
    y_pm = svm._convert_labels(y)
    n, d = X.shape

    # Analytic subgradient
    scores = X @ svm.w_ + svm.b_
    margin = y_pm * scores
    violating = margin < 1.0
    grad_w = svm.w_.copy()
    if violating.any():
        grad_w -= (svm.C / n) * (y_pm[violating, np.newaxis] * X[violating]).sum(axis=0)

    dims = rng.choice(d, size=min(n_check, d), replace=False)
    relative_errors = []

    for j in dims:
        w_plus = svm.w_.copy(); w_plus[j] += epsilon
        w_minus = svm.w_.copy(); w_minus[j] -= epsilon

        def obj(w):
            sc = X @ w + svm.b_
            h = np.maximum(0.0, 1.0 - y_pm * sc)
            return 0.5 * np.dot(w, w) + (svm.C / n) * h.sum()

        g_num = (obj(w_plus) - obj(w_minus)) / (2 * epsilon)
        g_ana = grad_w[j]
        rel_err = abs(g_num - g_ana) / (abs(g_ana) + abs(g_num) + 1e-10)
        relative_errors.append(rel_err)

    return {
        "max_relative_error": float(np.max(relative_errors)),
        "mean_relative_error": float(np.mean(relative_errors)),
        "passed": float(np.max(relative_errors)) < 0.01,
    }
