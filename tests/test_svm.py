"""
test_svm.py
===========
Unit tests for the from-scratch SVM implementation.

These tests use synthetic NUMERICAL data only (not audio) to verify
mathematical correctness of the optimization, gradient computation,
and prediction logic.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.svm_from_scratch import LinearSVMScratch, numerical_gradient_check


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _make_separable_data(n=200, d=10, random_state=0):
    """Generate linearly separable binary data."""
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n, d))
    w_true = rng.standard_normal(d)
    y = (np.sign(X @ w_true) + 1) // 2  # {0, 1}
    y = y.astype(int)
    return X.astype(np.float32), y


def _make_nonseparable_data(n=200, d=5, noise=0.3, random_state=42):
    """Generate noisy (non-separable) binary data."""
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + rng.normal(0, noise, n) > 0).astype(int)
    return X, y


# ─────────────────────────────────────────────
# Initialization tests
# ─────────────────────────────────────────────

class TestSVMInit:
    def test_default_params(self):
        svm = LinearSVMScratch()
        assert svm.C == 1.0
        assert svm.max_epochs == 100
        assert svm.batch_size == 32

    def test_custom_params(self):
        svm = LinearSVMScratch(C=10.0, max_epochs=50, batch_size=16)
        assert svm.C == 10.0
        assert svm.max_epochs == 50

    def test_not_fitted_raises(self):
        svm = LinearSVMScratch()
        X = np.random.randn(10, 5).astype(np.float32)
        with pytest.raises(RuntimeError):
            svm.predict(X)
        with pytest.raises(RuntimeError):
            svm.decision_function(X)


# ─────────────────────────────────────────────
# Label conversion tests
# ─────────────────────────────────────────────

class TestLabelConversion:
    def test_binary_01_to_pm1(self):
        y = np.array([0, 1, 0, 1])
        y_pm = LinearSVMScratch._convert_labels(y)
        assert set(y_pm) == {-1.0, 1.0}
        assert y_pm[0] == -1.0
        assert y_pm[1] == 1.0

    def test_already_pm1(self):
        y = np.array([-1, 1, -1, 1])
        y_pm = LinearSVMScratch._convert_labels(y)
        assert set(y_pm) == {-1.0, 1.0}

    def test_invalid_labels_raise(self):
        y = np.array([0, 2, 1])
        with pytest.raises(AssertionError):
            LinearSVMScratch._convert_labels(y)


# ─────────────────────────────────────────────
# Fit tests
# ─────────────────────────────────────────────

class TestSVMFit:
    def test_fit_returns_self(self):
        X, y = _make_separable_data(n=100, d=5)
        svm = LinearSVMScratch(max_epochs=5, verbose=False)
        result = svm.fit(X, y)
        assert result is svm

    def test_weights_nonzero_after_fit(self):
        X, y = _make_separable_data(n=100, d=5)
        svm = LinearSVMScratch(max_epochs=10, verbose=False)
        svm.fit(X, y)
        assert svm.w_ is not None
        assert np.linalg.norm(svm.w_) > 0

    def test_weight_dimension_matches_features(self):
        d = 15
        X, y = _make_separable_data(n=100, d=d)
        svm = LinearSVMScratch(max_epochs=5, verbose=False)
        svm.fit(X, y)
        assert len(svm.w_) == d

    def test_fitted_flag(self):
        X, y = _make_separable_data(n=50, d=5)
        svm = LinearSVMScratch(max_epochs=5, verbose=False)
        assert not svm.fitted_
        svm.fit(X, y)
        assert svm.fitted_

    def test_objective_history_recorded(self):
        X, y = _make_separable_data(n=100, d=5)
        svm = LinearSVMScratch(max_epochs=10, verbose=False)
        svm.fit(X, y)
        assert len(svm.objective_history_) > 0

    def test_objective_decreases(self):
        """Objective should generally decrease over training epochs."""
        X, y = _make_separable_data(n=200, d=10)
        svm = LinearSVMScratch(max_epochs=50, batch_size=32, verbose=False)
        svm.fit(X, y)
        # Compare first 5 epochs vs last 5 epochs
        early = np.mean(svm.objective_history_[:5])
        late = np.mean(svm.objective_history_[-5:])
        assert late < early, f"Expected objective to decrease: early={early:.4f}, late={late:.4f}"

    def test_training_time_recorded(self):
        X, y = _make_separable_data(n=50, d=5)
        svm = LinearSVMScratch(max_epochs=5, verbose=False)
        svm.fit(X, y)
        assert svm.training_time_ > 0


# ─────────────────────────────────────────────
# Predict tests
# ─────────────────────────────────────────────

class TestSVMPredict:
    def test_predict_returns_binary(self):
        X, y = _make_separable_data(n=100, d=5)
        svm = LinearSVMScratch(max_epochs=20, verbose=False)
        svm.fit(X, y)
        preds = svm.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_length_matches_input(self):
        X, y = _make_separable_data(n=100, d=5)
        svm = LinearSVMScratch(max_epochs=10, verbose=False)
        svm.fit(X, y)
        assert len(svm.predict(X)) == len(X)

    def test_decision_function_shape(self):
        X, y = _make_separable_data(n=50, d=5)
        svm = LinearSVMScratch(max_epochs=5, verbose=False)
        svm.fit(X, y)
        scores = svm.decision_function(X)
        assert scores.shape == (50,)

    def test_separable_data_high_accuracy(self):
        X, y = _make_separable_data(n=300, d=10, random_state=7)
        svm = LinearSVMScratch(C=1.0, max_epochs=100, batch_size=32, verbose=False)
        svm.fit(X, y)
        preds = svm.predict(X)
        acc = np.mean(preds == y)
        assert acc > 0.85, f"Expected high train accuracy, got {acc:.3f}"

    def test_predict_with_binary_01_labels(self):
        X, y = _make_nonseparable_data(n=200)
        svm = LinearSVMScratch(max_epochs=20, verbose=False)
        svm.fit(X, y)
        preds = svm.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})


# ─────────────────────────────────────────────
# Platt scaling prediction
# ─────────────────────────────────────────────

class TestPlattPrediction:
    def test_proba_shape(self):
        X, y = _make_separable_data(n=100, d=5)
        svm = LinearSVMScratch(max_epochs=10, verbose=False)
        svm.fit(X, y)
        proba = svm.predict_proba_platt(X, A=-1.0, B=0.0)
        assert proba.shape == (100, 2)

    def test_proba_sum_to_one(self):
        X, y = _make_separable_data(n=100, d=5)
        svm = LinearSVMScratch(max_epochs=10, verbose=False)
        svm.fit(X, y)
        proba = svm.predict_proba_platt(X, A=-1.0, B=0.0)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), \
            f"Probabilities don't sum to 1: {row_sums[:5]}"

    def test_proba_range(self):
        X, y = _make_separable_data(n=100, d=5)
        svm = LinearSVMScratch(max_epochs=10, verbose=False)
        svm.fit(X, y)
        proba = svm.predict_proba_platt(X, A=-1.0, B=0.0)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)


# ─────────────────────────────────────────────
# Gradient check
# ─────────────────────────────────────────────

class TestGradientCheck:
    def test_gradient_check_passes(self):
        X, y = _make_nonseparable_data(n=50, d=5)
        svm = LinearSVMScratch(C=1.0, max_epochs=5, verbose=False)
        svm.fit(X, y)
        result = numerical_gradient_check(svm, X, y, n_check=10)
        assert result["passed"], (
            f"Gradient check failed: max_rel_err={result['max_relative_error']:.6f}"
        )


# ─────────────────────────────────────────────
# Margin and support vector diagnostics
# ─────────────────────────────────────────────

class TestMarginAnalysis:
    def test_margin_violation_dict_keys(self):
        X, y = _make_separable_data(n=100, d=5)
        svm = LinearSVMScratch(max_epochs=10, verbose=False)
        svm.fit(X, y)
        result = svm.compute_margin_violations(X, y)
        assert "n_violations" in result
        assert "mean_hinge_loss" in result
        assert "n_outside_margin" in result

    def test_geometric_margin_positive(self):
        X, y = _make_separable_data(n=100, d=5)
        svm = LinearSVMScratch(max_epochs=10, verbose=False)
        svm.fit(X, y)
        margin = svm.geometric_margin()
        assert margin > 0

    def test_support_vectors_count(self):
        X, y = _make_separable_data(n=200, d=10)
        svm = LinearSVMScratch(C=1.0, max_epochs=50, verbose=False)
        svm.fit(X, y)
        assert svm.n_support_vectors_ >= 0
        assert svm.n_support_vectors_ <= len(X)


# ─────────────────────────────────────────────
# C regularization effect
# ─────────────────────────────────────────────

class TestRegularization:
    def test_large_C_smaller_margin(self):
        """
        Larger C → less regularisation → smaller geometric margin.
        """
        X, y = _make_nonseparable_data(n=200, d=5)
        svm_small = LinearSVMScratch(C=0.01, max_epochs=50, verbose=False)
        svm_large = LinearSVMScratch(C=100.0, max_epochs=50, verbose=False)
        svm_small.fit(X, y)
        svm_large.fit(X, y)
        # ||w|| should be larger for large C
        assert np.linalg.norm(svm_large.w_) >= np.linalg.norm(svm_small.w_)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
