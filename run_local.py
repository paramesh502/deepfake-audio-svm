#!/usr/bin/env python3
"""
run_local.py
=============
Local entry point for the Deepfake Audio Detection SVM project.

Runs the complete pipeline on your laptop:
  1. Load dataset (from a local directory)
  2. Extract features (MFCC, LFCC, spectral, prosodic)
  3. Train the from-scratch SVM + sklearn baselines
  4. Evaluate on held-out test set
  5. Save the trained model pipeline
  6. Optionally launch the Gradio web app

Usage
-----
  python run_local.py --data-dir ./data --mode quick
  python run_local.py --data-dir ./data --mode full --launch-app
  python run_local.py --data-dir ./data --quick   # shorthand

This is the local equivalent of the Colab notebook.
It does NOT require Google Drive, Google Colab, or any cloud service.
"""

import os
import sys
import argparse
import time
import json
import logging

# Allow running from project root without installation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Deepfake Audio Detection SVM — Local Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-dir", default="./data",
        help="Path to the dataset directory (default: ./data)",
    )
    p.add_argument(
        "--model-dir", default="./models",
        help="Where to save trained model artifacts (default: ./models)",
    )
    p.add_argument(
        "--results-dir", default="./results",
        help="Where to save figures and metrics (default: ./results)",
    )
    p.add_argument(
        "--mode", choices=["quick", "full"], default="quick",
        help="quick = small subset (250/class), full = all available data (default: quick)",
    )
    p.add_argument(
        "--max-per-class", type=int, default=None,
        help="Override max samples per class (None = all available)",
    )
    p.add_argument(
        "--external-dir", default=None,
        help="Path to external dataset for generalization testing (optional)",
    )
    p.add_argument(
        "--launch-app", action="store_true",
        help="Launch the Gradio web app after training completes",
    )
    p.add_argument(
        "--skip-training", action="store_true",
        help="Skip training; only launch the app using an existing saved model",
    )
    p.add_argument(
        "--sr", type=int, default=16000,
        help="Target sample rate in Hz (default: 16000)",
    )
    p.add_argument(
        "--epochs", type=int, default=100,
        help="Max epochs for from-scratch SVM (default: 100)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────
# Environment detection
# ─────────────────────────────────────────────

def detect_environment():
    """Return 'colab' or 'local'."""
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        return "local"


# ─────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────

def parse_asvspoof_protocol(protocol_path):
    """Parse ASVspoof 2019 LA protocol file → {filename: label}."""
    labels = {}
    if not os.path.exists(protocol_path):
        logger.warning(f"Protocol file not found: {protocol_path}")
        return labels
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels[parts[1]] = 0 if parts[-1] == "bonafide" else 1
    return labels


def build_file_list(audio_dir, label_dict, max_per_class=None):
    """Build (file_paths, labels) from a directory + label dict."""
    files, labels = [], []
    if not os.path.isdir(audio_dir):
        logger.warning(f"Audio directory not found: {audio_dir}")
        return files, labels
    real_count, fake_count = 0, 0
    for fname in sorted(os.listdir(audio_dir)):
        if not (fname.endswith(".flac") or fname.endswith(".wav")):
            continue
        key = fname.replace(".flac", "").replace(".wav", "")
        if key not in label_dict:
            continue
        lab = label_dict[key]
        if max_per_class:
            if lab == 0 and real_count >= max_per_class:
                continue
            if lab == 1 and fake_count >= max_per_class:
                continue
        files.append(os.path.join(audio_dir, fname))
        labels.append(lab)
        if lab == 0:
            real_count += 1
        else:
            fake_count += 1
    return files, labels


def load_dataset(data_dir, mode="quick", max_per_class=None):
    """
    Load ASVspoof 2019 LA dataset from a local directory.

    Expected structure:
        data_dir/LA/ASVspoof2019_LA_train/flac/
        data_dir/LA/ASVspoof2019_LA_dev/flac/
        data_dir/LA/ASVspoof2019_LA_eval/flac/
        data_dir/LA/ASVspoof2019_LA_cm_protocols/
    """
    asvspoof_dir = os.path.join(data_dir, "LA")
    protocol_dir = os.path.join(asvspoof_dir, "ASVspoof2019_LA_cm_protocols")

    if max_per_class is None:
        max_per_class = 250 if mode == "quick" else None

    train_proto = os.path.join(protocol_dir, "ASVspoof2019.LA.cm.train.trn.txt")
    dev_proto = os.path.join(protocol_dir, "ASVspoof2019.LA.cm.dev.ndx.txt")
    eval_proto = os.path.join(protocol_dir, "ASVspoof2019.LA.cm.eval.trl.txt")

    train_labels_dict = parse_asvspoof_protocol(train_proto)
    dev_labels_dict = parse_asvspoof_protocol(dev_proto)
    eval_labels_dict = parse_asvspoof_protocol(eval_proto)

    train_files, train_labels = build_file_list(
        os.path.join(asvspoof_dir, "ASVspoof2019_LA_train", "flac"),
        train_labels_dict, max_per_class,
    )
    dev_files, dev_labels = build_file_list(
        os.path.join(asvspoof_dir, "ASVspoof2019_LA_dev", "flac"),
        dev_labels_dict, max_per_class,
    )
    eval_files, eval_labels = build_file_list(
        os.path.join(asvspoof_dir, "ASVspoof2019_LA_eval", "flac"),
        eval_labels_dict, max_per_class,
    )

    return (train_files, train_labels), (dev_files, dev_labels), (eval_files, eval_labels)


# ─────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────

def extract_features_for_files(files, labels, sr=16000, cache_path=None):
    """Extract features for all files. Uses cache if available."""
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"]

    from src.features import extract_all_features, FeatureConfig
    from src.preprocessing import load_audio, normalize_waveform

    cfg = FeatureConfig(sr=sr)
    feats, valid_labels = [], []

    for i, (f, l) in enumerate(zip(files, labels)):
        if not os.path.exists(f):
            continue
        try:
            y, _ = load_audio(f, target_sr=sr)
            y = normalize_waveform(y, method="peak")
            feat = extract_all_features(y, cfg)
            feats.append(feat)
            valid_labels.append(l)
        except Exception as e:
            logger.warning(f"Skip {os.path.basename(f)}: {e}")
        if (i + 1) % 200 == 0:
            logger.info(f"  Feature extraction: {i+1}/{len(files)}")

    X = np.array(feats, dtype=np.float32)
    y = np.array(valid_labels, dtype=int)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y)
        logger.info(f"Features cached to {cache_path}")

    return X, y


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    env = detect_environment()
    logger.info(f"Environment: {env} | Mode: {args.mode}")

    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    for sub in ["figures", "tables", "metrics"]:
        os.makedirs(os.path.join(args.results_dir, sub), exist_ok=True)

    # ── Skip training mode: just launch the app ──
    if args.skip_training:
        logger.info("Skipping training. Launching app with existing model.")
        model_path = os.path.join(args.model_dir, "inference_pipeline.joblib")
        if not os.path.exists(model_path):
            logger.error(f"No saved model found at {model_path}. Run without --skip-training first.")
            sys.exit(1)
        launch_app(model_path)
        return

    # ── Load dataset ──
    logger.info(f"Loading dataset from {args.data_dir}")
    (train_files, train_labels), (dev_files, dev_labels), (eval_files, eval_labels) = \
        load_dataset(args.data_dir, mode=args.mode, max_per_class=args.max_per_class)

    logger.info(f"Train: {len(train_files)} ({train_labels.count(0)} real, {train_labels.count(1)} fake)")
    logger.info(f"Dev:   {len(dev_files)} ({dev_labels.count(0)} real, {dev_labels.count(1)} fake)")
    logger.info(f"Eval:  {len(eval_files)} ({eval_labels.count(0)} real, {eval_labels.count(1)} fake)")

    if len(train_files) == 0:
        logger.error(
            "No training files found. Please download ASVspoof 2019 LA and place it in:\n"
            f"  {args.data_dir}/LA/ASVspoof2019_LA_train/flac/\n"
            f"  {args.data_dir}/LA/ASVspoof2019_LA_cm_protocols/\n"
            "See documentation/dataset_instructions.md for download instructions."
        )
        sys.exit(1)

    # ── Extract features ──
    logger.info("Extracting features (this may take a while)...")
    cache_dir = os.path.join(args.data_dir, "features_cache")
    os.makedirs(cache_dir, exist_ok=True)

    X_train, y_train = extract_features_for_files(
        train_files, train_labels, sr=args.sr,
        cache_path=os.path.join(cache_dir, "train.npz"),
    )
    X_dev, y_dev = extract_features_for_files(
        dev_files, dev_labels, sr=args.sr,
        cache_path=os.path.join(cache_dir, "dev.npz"),
    )
    X_eval, y_eval = extract_features_for_files(
        eval_files, eval_labels, sr=args.sr,
        cache_path=os.path.join(cache_dir, "eval.npz"),
    )
    logger.info(f"Feature dimension: {X_train.shape[1]}")

    # ── Train models ──
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.svm import SVC, LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from src.svm_from_scratch import LinearSVMScratch, numerical_gradient_check
    from src.calibration import PlattCalibrator

    # Split off calibration set
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    # Preprocessing (fit on train only)
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_fit_sc = scaler.fit_transform(imputer.fit_transform(X_fit))
    X_dev_sc = scaler.transform(imputer.transform(X_dev))
    X_eval_sc = scaler.transform(imputer.transform(X_eval))
    X_cal_sc = scaler.transform(imputer.transform(X_cal))

    # 1. From-scratch SVM
    logger.info("=== Training from-scratch Linear SVM ===")
    scratch_svm = LinearSVMScratch(C=1.0, max_epochs=args.epochs, batch_size=64, verbose=True)
    scratch_svm.fit(X_fit_sc, y_fit)
    logger.info(f"Scratch SVM: {scratch_svm.training_time_:.2f}s, "
                f"support vectors ≈ {scratch_svm.n_support_vectors_}")

    # Gradient check
    grad_check = numerical_gradient_check(scratch_svm, X_fit_sc[:200], y_fit[:200])
    logger.info(f"Gradient check: max_rel_err={grad_check['max_relative_error']:.6f} "
                f"({'PASS' if grad_check['passed'] else 'FAIL'})")

    # 2. sklearn SVC(RBF) — the deployed model
    logger.info("=== Training sklearn SVC(RBF) ===")
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True)),
    ])
    param_grid = {
        "clf__C": [0.1, 1, 10],
        "clf__gamma": ["scale", 0.01],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=0)
    gs.fit(X_fit, y_fit)
    best_sklearn = gs.best_estimator_
    logger.info(f"Best SVC(RBF): {gs.best_params_}  CV F1: {gs.best_score_:.4f}")

    # 3. Calibration (Platt scaling for scratch SVM)
    logger.info("=== Fitting Platt calibrator ===")
    cal_scores = scratch_svm.decision_function(X_cal_sc)
    platt = PlattCalibrator()
    platt.fit(cal_scores, y_cal)

    # ── Evaluate ──
    logger.info("=== Evaluation ===")

    # Scratch SVM on dev
    y_pred_dev = scratch_svm.predict(X_dev_sc)
    dev_acc = accuracy_score(y_dev, y_pred_dev)
    dev_f1 = f1_score(y_dev, y_pred_dev)
    logger.info(f"Scratch SVM — Dev: acc={dev_acc:.4f} f1={dev_f1:.4f}")

    # sklearn on dev
    y_pred_dev_sk = best_sklearn.predict(X_dev)
    dev_acc_sk = accuracy_score(y_dev, y_pred_dev_sk)
    dev_f1_sk = f1_score(y_dev, y_pred_dev_sk)
    logger.info(f"SVC(RBF) — Dev: acc={dev_acc_sk:.4f} f1={dev_f1_sk:.4f}")

    # Test set evaluation
    if len(X_eval) > 0:
        y_pred_eval = scratch_svm.predict(X_eval_sc)
        eval_acc = accuracy_score(y_eval, y_pred_eval)
        eval_f1 = f1_score(y_eval, y_pred_eval)
        cm = confusion_matrix(y_eval, y_pred_eval)
        logger.info(f"Scratch SVM — Eval: acc={eval_acc:.4f} f1={eval_f1:.4f}")
        logger.info(f"Confusion matrix:\n{cm}")

    # ── Save model ──
    import joblib

    # Choose the final deployed model (best validation F1)
    if dev_f1_sk >= dev_f1:
        final_model = best_sklearn
        model_type = "sklearn_svc_rbf"
        logger.info("Deployed model: sklearn SVC(RBF)")
    else:
        final_model = None
        model_type = "scratch_svm"
        logger.info("Deployed model: scratch linear SVM")

    bundle = {
        "model_type": model_type,
        "model": final_model,
        "scratch_w": scratch_svm.w_,
        "scratch_b": scratch_svm.b_,
        "scratch_C": scratch_svm.C,
        "imputer": imputer,
        "scaler": scaler,
        "calibrator": {"A": platt.A, "B": platt.B},
        "feature_config": {
            "sr": args.sr,
            "n_mfcc": 20, "n_lfcc": 20,
            "use_mfcc": True, "use_lfcc": True,
            "use_spectral": True, "use_prosodic": True,
            "use_mfcc_delta": True, "use_mfcc_delta2": True,
            "use_lfcc_delta": True, "use_lfcc_delta2": True,
            "use_chroma": False,
            "n_spectral_contrast_bands": 6,
            "n_chroma": 12,
            "stat_functions": ["mean", "std", "median", "p25", "p75", "max"],
        },
        "metadata": {
            "dataset": "ASVspoof 2019 LA",
            "mode": args.mode,
            "dev_f1": float(dev_f1),
            "dev_f1_sklearn": float(dev_f1_sk),
            "gradient_check_passed": grad_check["passed"],
        },
    }
    model_path = os.path.join(args.model_dir, "inference_pipeline.joblib")
    joblib.dump(bundle, model_path, compress=3)
    logger.info(f"Model saved to {model_path}")

    # Save metrics
    metrics = {
        "scratch_svm": {"dev_acc": float(dev_acc), "dev_f1": float(dev_f1)},
        "sklearn_svc_rbf": {"dev_acc": float(dev_acc_sk), "dev_f1": float(dev_f1_sk),
                            "best_params": gs.best_params_},
        "gradient_check": grad_check,
    }
    with open(os.path.join(args.results_dir, "metrics", "local_run_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to {args.results_dir}/metrics/")

    logger.info("\n✅ Pipeline complete!")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Results: {args.results_dir}/")

    # ── Launch app ──
    if args.launch_app:
        launch_app(model_path)


# ─────────────────────────────────────────────
# App launcher
# ─────────────────────────────────────────────

def launch_app(model_path):
    """Launch the Gradio web app using a saved model."""
    try:
        import joblib
        from src.app import build_app
        import gradio as gr
    except ImportError as e:
        logger.error(f"Cannot launch app: {e}")
        logger.error("Install gradio: pip install gradio")
        return

    logger.info(f"Loading model from {model_path}...")
    bundle = joblib.load(model_path)

    # Build a lightweight wrapper that the app can use
    from src.features import FeatureConfig
    from src.inference import InferencePipeline
    from src.calibration import PlattCalibrator

    cfg = FeatureConfig(**{k: v for k, v in bundle["feature_config"].items()
                          if hasattr(FeatureConfig(), k)})

    calibrator = None
    if bundle.get("calibrator"):
        calibrator = PlattCalibrator.from_dict(bundle["calibrator"])

    if bundle["model_type"] == "scratch_svm":
        # Reconstruct the scratch SVM for inference
        from src.svm_from_scratch import LinearSVMScratch
        svm = LinearSVMScratch(C=bundle["scratch_C"], verbose=False)
        svm.w_ = bundle["scratch_w"]
        svm.b_ = bundle["scratch_b"]
        svm.fitted_ = True
        model = svm
    else:
        model = bundle["model"]

    pipeline = InferencePipeline(
        model=model,
        cfg=cfg,
        model_name=bundle["model_type"],
        metadata=bundle.get("metadata", {}),
        scaler=bundle.get("scaler"),
        imputer=bundle.get("imputer"),
        calibrator=calibrator,
    )

    logger.info("Launching EchoVerify AI web app...")
    demo = build_app(pipeline)
    demo.launch(share=False, debug=True)


if __name__ == "__main__":
    main()
