"""
model_training.py
=================
Training orchestration for all SVM variants and baselines.

Models trained
--------------
1. LinearSVMScratch        — from-scratch NumPy implementation
2. scikit-learn LinearSVC  — LIBLINEAR-based fast linear SVM
3. scikit-learn SVC(linear)— kernel SVM with linear kernel
4. scikit-learn SVC(RBF)   — kernel SVM with RBF kernel (nonlinear)
5. LogisticRegression      — baseline
6. RandomForestClassifier  — ensemble baseline

All sklearn models are wrapped in a Pipeline:
    SimpleImputer → StandardScaler → [optional PCA] → classifier

Hyperparameter tuning is done with GridSearchCV using the development
(validation) partition, never the held-out test set.
"""

import time
import logging
import warnings
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

from .svm_from_scratch import LinearSVMScratch

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────
# Preprocessing pipeline builder
# ─────────────────────────────────────────────

def make_preprocessing_pipeline(
    use_pca: bool = False,
    pca_variance: float = 0.95,
) -> List:
    """
    Return the common preprocessing steps as a list of (name, estimator) tuples.

    Imputation is included to handle any residual NaN values after feature
    extraction (defensive practice).
    """
    steps = [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_variance, svd_solver="full")))
    return steps


# ─────────────────────────────────────────────
# Individual model builders
# ─────────────────────────────────────────────

def build_sklearn_linear_svc(use_pca: bool = False) -> Pipeline:
    steps = make_preprocessing_pipeline(use_pca=use_pca)
    steps.append(("clf", LinearSVC(max_iter=10_000, dual=False)))
    return Pipeline(steps)


def build_sklearn_svc_linear(use_pca: bool = False) -> Pipeline:
    steps = make_preprocessing_pipeline(use_pca=use_pca)
    steps.append(("clf", SVC(kernel="linear", probability=True, max_iter=10_000)))
    return Pipeline(steps)


def build_sklearn_svc_rbf(use_pca: bool = False) -> Pipeline:
    steps = make_preprocessing_pipeline(use_pca=use_pca)
    steps.append(("clf", SVC(kernel="rbf", probability=True)))
    return Pipeline(steps)


def build_logistic_regression(use_pca: bool = False) -> Pipeline:
    steps = make_preprocessing_pipeline(use_pca=use_pca)
    steps.append(("clf", LogisticRegression(max_iter=5000, solver="lbfgs")))
    return Pipeline(steps)


def build_random_forest(use_pca: bool = False) -> Pipeline:
    steps = make_preprocessing_pipeline(use_pca=use_pca)
    steps.append(("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)))
    return Pipeline(steps)


# ─────────────────────────────────────────────
# Hyperparameter grids
# ─────────────────────────────────────────────

PARAM_GRIDS = {
    "linear_svc": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
    },
    "svc_linear": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
    },
    "svc_rbf": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__gamma": ["scale", 0.001, 0.01, 0.1],
    },
    "logistic": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
    },
}

SCRATCH_SVM_C_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0]


# ─────────────────────────────────────────────
# Training + tuning functions
# ─────────────────────────────────────────────

def tune_sklearn_pipeline(
    pipeline: Pipeline,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 3,
    scoring: str = "f1",
    n_jobs: int = -1,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Run GridSearchCV on a pipeline and return the best fitted pipeline + results.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
        return_train_score=True,
    )
    t0 = time.perf_counter()
    gs.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    results = {
        "best_params": gs.best_params_,
        "best_cv_score": gs.best_score_,
        "cv_results": gs.cv_results_,
        "training_time_s": elapsed,
    }
    logger.info(
        f"Best params: {gs.best_params_}  "
        f"CV {scoring}: {gs.best_score_:.4f}  "
        f"Time: {elapsed:.1f}s"
    )
    return gs.best_estimator_, results


def tune_scratch_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C_candidates: List[float] = None,
    max_epochs: int = 50,
    batch_size: int = 64,
    random_state: int = 42,
) -> Tuple["LinearSVMScratch", np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Manual hyperparameter search for the scratch SVM.

    Uses a separate validation set so there is no data leakage.
    Feature scaling is applied using parameters fitted on X_train only.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from .evaluation import compute_metrics

    if C_candidates is None:
        C_candidates = SCRATCH_SVM_C_CANDIDATES

    # Fit preprocessing on training data only
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    X_train_sc = scaler.fit_transform(imputer.fit_transform(X_train))
    X_val_sc = scaler.transform(imputer.transform(X_val))

    best_f1 = -1.0
    best_C = C_candidates[0]
    best_svm = None
    search_results = []

    for C in C_candidates:
        svm = LinearSVMScratch(
            C=C,
            max_epochs=max_epochs,
            batch_size=batch_size,
            random_state=random_state,
            verbose=False,
        )
        svm.fit(X_train_sc, y_train)
        y_pred_val = svm.predict(X_val_sc)
        metrics = compute_metrics(y_val, y_pred_val)
        f1 = metrics["f1"]

        logger.info(f"  C={C:.4f}  val_f1={f1:.4f}")
        search_results.append({"C": C, "val_f1": f1, "val_acc": metrics["accuracy"]})

        if f1 > best_f1:
            best_f1 = f1
            best_C = C
            best_svm = svm

    logger.info(f"Best scratch SVM: C={best_C}  val_f1={best_f1:.4f}")

    tuning_info = {
        "best_C": best_C,
        "best_val_f1": best_f1,
        "search_results": search_results,
        "imputer": imputer,
        "scaler": scaler,
    }
    return best_svm, X_train_sc, X_val_sc, tuning_info


# ─────────────────────────────────────────────
# Full training orchestration
# ─────────────────────────────────────────────

def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scratch_max_epochs: int = 50,
    scratch_batch_size: int = 64,
    cv_folds: int = 3,
    use_pca: bool = False,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train all models and return a results dictionary.

    Returns
    -------
    dict with model names as keys, each value containing:
        model       : fitted estimator
        metrics_val : dict of validation metrics
        tuning      : dict of tuning results
        time_s      : training time
    """
    from .evaluation import compute_metrics

    results = {}

    # ── 1. Scratch SVM ────────────────────────────────
    logger.info("=== Training Scratch Linear SVM ===")
    scratch_svm, X_tr_sc, X_val_sc, tuning_scratch = tune_scratch_svm(
        X_train, y_train, X_val, y_val,
        max_epochs=scratch_max_epochs,
        batch_size=scratch_batch_size,
        random_state=random_state,
    )
    y_pred_scratch = scratch_svm.predict(X_val_sc)
    metrics_scratch = compute_metrics(y_val, y_pred_scratch)

    results["scratch_svm"] = {
        "model": scratch_svm,
        "imputer": tuning_scratch["imputer"],
        "scaler": tuning_scratch["scaler"],
        "metrics_val": metrics_scratch,
        "tuning": tuning_scratch,
        "time_s": scratch_svm.training_time_,
    }

    # ── 2. sklearn LinearSVC ──────────────────────────
    logger.info("=== Training sklearn LinearSVC ===")
    pipe_lsvc = build_sklearn_linear_svc(use_pca=use_pca)
    best_lsvc, tune_lsvc = tune_sklearn_pipeline(
        pipe_lsvc, PARAM_GRIDS["linear_svc"],
        X_train, y_train, cv=cv_folds, n_jobs=n_jobs,
    )
    y_pred_lsvc = best_lsvc.predict(X_val)
    results["linear_svc"] = {
        "model": best_lsvc,
        "metrics_val": compute_metrics(y_val, y_pred_lsvc),
        "tuning": tune_lsvc,
        "time_s": tune_lsvc["training_time_s"],
    }

    # ── 3. sklearn SVC(linear) ───────────────────────
    logger.info("=== Training sklearn SVC (linear kernel) ===")
    pipe_svcl = build_sklearn_svc_linear(use_pca=use_pca)
    best_svcl, tune_svcl = tune_sklearn_pipeline(
        pipe_svcl, PARAM_GRIDS["svc_linear"],
        X_train, y_train, cv=cv_folds, n_jobs=n_jobs,
    )
    y_pred_svcl = best_svcl.predict(X_val)
    results["svc_linear"] = {
        "model": best_svcl,
        "metrics_val": compute_metrics(y_val, y_pred_svcl),
        "tuning": tune_svcl,
        "time_s": tune_svcl["training_time_s"],
    }

    # ── 4. sklearn SVC(RBF) ──────────────────────────
    logger.info("=== Training sklearn SVC (RBF kernel) ===")
    pipe_svcr = build_sklearn_svc_rbf(use_pca=use_pca)
    best_svcr, tune_svcr = tune_sklearn_pipeline(
        pipe_svcr, PARAM_GRIDS["svc_rbf"],
        X_train, y_train, cv=cv_folds, n_jobs=n_jobs,
    )
    y_pred_svcr = best_svcr.predict(X_val)
    results["svc_rbf"] = {
        "model": best_svcr,
        "metrics_val": compute_metrics(y_val, y_pred_svcr),
        "tuning": tune_svcr,
        "time_s": tune_svcr["training_time_s"],
    }

    # ── 5. Logistic Regression ────────────────────────
    logger.info("=== Training Logistic Regression ===")
    pipe_lr = build_logistic_regression(use_pca=use_pca)
    best_lr, tune_lr = tune_sklearn_pipeline(
        pipe_lr, PARAM_GRIDS["logistic"],
        X_train, y_train, cv=cv_folds, n_jobs=n_jobs,
    )
    y_pred_lr = best_lr.predict(X_val)
    results["logistic_regression"] = {
        "model": best_lr,
        "metrics_val": compute_metrics(y_val, y_pred_lr),
        "tuning": tune_lr,
        "time_s": tune_lr["training_time_s"],
    }

    # ── 6. Random Forest ─────────────────────────────
    logger.info("=== Training Random Forest ===")
    t0 = time.perf_counter()
    pipe_rf = build_random_forest(use_pca=use_pca)
    pipe_rf.fit(X_train, y_train)
    rf_time = time.perf_counter() - t0
    y_pred_rf = pipe_rf.predict(X_val)
    results["random_forest"] = {
        "model": pipe_rf,
        "metrics_val": compute_metrics(y_val, y_pred_rf),
        "tuning": {},
        "time_s": rf_time,
    }

    return results


def select_best_model(
    results: Dict[str, Any],
    criterion: str = "f1",
) -> str:
    """
    Select the best model from training results based on a validation criterion.

    'criterion' can be any key in the metrics_val dicts.
    """
    best_name = None
    best_score = -np.inf

    for name, info in results.items():
        score = info["metrics_val"].get(criterion, -np.inf)
        if score > best_score:
            best_score = score
            best_name = name

    logger.info(f"Best model: {best_name}  (val {criterion} = {best_score:.4f})")
    return best_name
