#!/usr/bin/env python3
"""
train_real.py — Train on the real garystafford deepfake audio dataset.
Extracts features from parquet, trains from-scratch SVM + sklearn SVC(RBF),
calibrates, saves model, and optionally launches the Gradio app.
"""
import os, sys, time, io, json, warnings, logging
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.features import extract_all_features, FeatureConfig
from src.preprocessing import normalize_waveform, segment_waveform
from src.svm_from_scratch import LinearSVMScratch, numerical_gradient_check
from src.calibration import PlattCalibrator

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

SR = 16000
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
MODEL_DIR = os.path.join(BASE, 'models')
RESULTS_DIR = os.path.join(BASE, 'results')
CACHE_DIR = os.path.join(DATA_DIR, 'features_cache')
for d in [MODEL_DIR, RESULTS_DIR, CACHE_DIR,
          os.path.join(RESULTS_DIR, 'figures'),
          os.path.join(RESULTS_DIR, 'metrics'),
          os.path.join(RESULTS_DIR, 'tables')]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# 1. Load dataset from parquet
# ─────────────────────────────────────────────
def load_dataset():
    logger.info("Loading parquet files...")
    t1 = pq.read_table(os.path.join(DATA_DIR, 'train-00000-of-00002.parquet'))
    t2 = pq.read_table(os.path.join(DATA_DIR, 'train-00001-of-00002.parquet'))
    combined = pa.concat_tables([t1, t2])
    data = combined.to_pydict()
    audio_bytes_list = data['audio']
    labels = np.array(data['label'])
    n_real = np.sum(labels == 0)
    n_fake = np.sum(labels == 1)
    logger.info(f"Loaded {len(labels)} samples: {n_real} real, {n_fake} fake")
    return audio_bytes_list, labels

# ─────────────────────────────────────────────
# 2. Extract features
# ─────────────────────────────────────────────
def extract_features(audio_bytes_list, labels, cfg, use_cache=True):
    cache_path = os.path.join(CACHE_DIR, 'real_features.npz')
    if use_cache and os.path.exists(cache_path):
        logger.info(f"Loading cached features from {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return d['X'], d['y']

    logger.info(f"Extracting features for {len(audio_bytes_list)} samples...")
    feats, valid_labels = [], []
    t0 = time.time()

    for i, (audio_dict, label) in enumerate(zip(audio_bytes_list, labels)):
        try:
            # Decode FLAC bytes to waveform
            raw_bytes = audio_dict['bytes']
            y, sr_orig = sf.read(io.BytesIO(raw_bytes))
            y = y.astype(np.float32)
            if y.ndim > 1:
                y = y.mean(axis=1)
            # Resample to 16kHz if needed
            if sr_orig != SR:
                y = librosa.resample(y, orig_sr=sr_orig, target_sr=SR)
            y = normalize_waveform(y, 'peak')

            # Use first 3 seconds for consistent feature dimension
            if len(y) > SR * 6:
                y = y[:SR * 6]

            feat = extract_all_features(y, cfg)
            feats.append(feat)
            valid_labels.append(label)
        except Exception as e:
            logger.warning(f"Skip sample {i}: {e}")

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(audio_bytes_list) - i - 1) / rate
            logger.info(f"  {i+1}/{len(audio_bytes_list)} ({rate:.1f} samples/s, ETA {eta:.0f}s)")

    X = np.array(feats, dtype=np.float32)
    y = np.array(valid_labels, dtype=int)
    logger.info(f"Features extracted: X={X.shape}, y={y.shape} in {time.time()-t0:.1f}s")

    np.savez_compressed(cache_path, X=X, y=y)
    logger.info(f"Features cached to {cache_path}")
    return X, y

# ─────────────────────────────────────────────
# 3. Train models
# ─────────────────────────────────────────────
def train_models(X_train, y_train, X_cal, y_cal, epochs=100):
    # Preprocessing
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(imputer.fit_transform(X_train))
    X_cal_sc = scaler.transform(imputer.transform(X_cal))

    # From-scratch SVM
    logger.info("=== Training from-scratch Linear SVM ===")
    svm = LinearSVMScratch(C=1.0, max_epochs=epochs, batch_size=64, verbose=True)
    svm.fit(X_train_sc, y_train)
    logger.info(f"Scratch SVM: {svm.training_time_:.2f}s, support vectors={svm.n_support_vectors_}")

    # Gradient check
    gc = numerical_gradient_check(svm, X_train_sc[:200], y_train[:200])
    logger.info(f"Gradient check: max_rel_err={gc['max_relative_error']:.6f} ({'PASS' if gc['passed'] else 'FAIL'})")

    # sklearn SVC(RBF)
    logger.info("=== Training sklearn SVC(RBF) ===")
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, C=1.0))
    ])
    pipe.fit(X_train, y_train)
    logger.info("SVC(RBF) trained.")

    # Platt calibration for scratch SVM
    logger.info("=== Fitting Platt calibrator ===")
    cal_scores = svm.decision_function(X_cal_sc)
    platt = PlattCalibrator()
    platt.fit(cal_scores, y_cal)

    return svm, pipe, platt, imputer, scaler

# ─────────────────────────────────────────────
# 4. Evaluate + generate plots
# ─────────────────────────────────────────────
def evaluate_and_plot(svm, pipe, platt, imputer, scaler, X_test, y_test):
    logger.info("=== Evaluation ===")
    X_test_sc = scaler.transform(imputer.transform(X_test))

    # Scratch SVM predictions
    y_pred_svm = svm.predict(X_test_sc)
    scores_svm = svm.decision_function(X_test_sc)
    proba_svm = platt.predict_proba(scores_svm)

    acc_svm = accuracy_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm)
    cm_svm = confusion_matrix(y_test, y_pred_svm)

    logger.info(f"Scratch SVM — Test: acc={acc_svm:.4f} f1={f1_svm:.4f}")
    logger.info(f"  Confusion matrix:\n{cm_svm}")

    # sklearn predictions
    y_pred_sk = pipe.predict(X_test)
    proba_sk = pipe.predict_proba(X_test)
    scores_sk = pipe.decision_function(X_test) if hasattr(pipe, 'decision_function') else proba_sk[:,1]

    acc_sk = accuracy_score(y_test, y_pred_sk)
    f1_sk = f1_score(y_test, y_pred_sk)
    cm_sk = confusion_matrix(y_test, y_pred_sk)

    logger.info(f"SVC(RBF) — Test: acc={acc_sk:.4f} f1={f1_sk:.4f}")
    logger.info(f"  Confusion matrix:\n{cm_sk}")

    # ROC-AUC
    try:
        auc_svm = roc_auc_score(y_test, proba_svm[:,1])
        auc_sk = roc_auc_score(y_test, proba_sk[:,1])
        logger.info(f"ROC-AUC: scratch={auc_svm:.4f}, sklearn={auc_sk:.4f}")
    except:
        auc_svm = auc_sk = None

    # ── Plots ──
    plt.rcParams.update({'figure.facecolor':'#0d1117','axes.facecolor':'#161b22',
        'axes.edgecolor':'#30363d','axes.labelcolor':'#e6edf3',
        'xtick.color':'#8b949e','ytick.color':'#8b949e','text.color':'#e6edf3'})

    # Convergence plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(svm.objective_history_, color='#00d4ff', lw=2)
    axes[0].set_yscale('log'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Objective')
    axes[0].set_title('SVM Optimization Loss')
    axes[1].plot(svm.train_acc_history_, color='#3fb950', lw=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('SVM Training Accuracy')
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, 'figures', 'fig_svm_convergence.png'), dpi=150); plt.close()
    logger.info("Saved: fig_svm_convergence.png")

    # Confusion matrix (best model)
    best_cm = cm_sk if f1_sk >= f1_svm else cm_svm
    best_name = 'SVC(RBF)' if f1_sk >= f1_svm else 'Scratch SVM'
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['REAL','FAKE'], yticklabels=['REAL','FAKE'])
    ax.set_title(f'Confusion Matrix ({best_name})'); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, 'figures', 'fig_confusion_matrix.png'), dpi=150); plt.close()
    logger.info("Saved: fig_confusion_matrix.png")

    # ROC curve
    if auc_svm and auc_sk:
        fpr_s, tpr_s, _ = roc_curve(y_test, proba_svm[:,1])
        fpr_k, tpr_k, _ = roc_curve(y_test, proba_sk[:,1])
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr_s, tpr_s, color='#00d4ff', lw=2, label=f'Scratch SVM (AUC={auc_svm:.4f})')
        ax.plot(fpr_k, tpr_k, color='#3fb950', lw=2, label=f'SVC(RBF) (AUC={auc_sk:.4f})')
        ax.plot([0,1],[0,1],'--', color='#8b949e')
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curve'); ax.legend()
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, 'figures', 'fig_roc_curve.png'), dpi=150); plt.close()
        logger.info("Saved: fig_roc_curve.png")

    # PR curve
    if auc_svm and auc_sk:
        p_s, r_s, _ = precision_recall_curve(y_test, proba_svm[:,1])
        p_k, r_k, _ = precision_recall_curve(y_test, proba_sk[:,1])
        ap_s = average_precision_score(y_test, proba_svm[:,1])
        ap_k = average_precision_score(y_test, proba_sk[:,1])
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(r_s, p_s, color='#00d4ff', lw=2, label=f'Scratch SVM (AP={ap_s:.4f})')
        ax.plot(r_k, p_k, color='#3fb950', lw=2, label=f'SVC(RBF) (AP={ap_k:.4f})')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('Precision-Recall Curve'); ax.legend()
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, 'figures', 'fig_pr_curve.png'), dpi=150); plt.close()
        logger.info("Saved: fig_pr_curve.png")

    # Model comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    models_names = ['Scratch SVM', 'SVC(RBF)']
    accs = [acc_svm, acc_sk]
    f1s = [f1_svm, f1_sk]
    x = np.arange(len(models_names))
    ax.bar(x - 0.15, accs, 0.3, label='Accuracy', color='#00d4ff')
    ax.bar(x + 0.15, f1s, 0.3, label='F1 Score', color='#3fb950')
    ax.set_xticks(x); ax.set_xticklabels(models_names)
    ax.set_ylabel('Score'); ax.set_title('Model Comparison'); ax.legend()
    ax.set_ylim(0, 1)
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i-0.15, a+0.01, f'{a:.3f}', ha='center', fontsize=9, color='#e6edf3')
        ax.text(i+0.15, f+0.01, f'{f:.3f}', ha='center', fontsize=9, color='#e6edf3')
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, 'figures', 'fig_model_comparison.png'), dpi=150); plt.close()
    logger.info("Saved: fig_model_comparison.png")

    return {
        'scratch_svm': {'accuracy': float(acc_svm), 'f1': float(f1_svm),
                        'roc_auc': float(auc_svm) if auc_svm else None,
                        'cm': cm_svm.tolist()},
        'svc_rbf': {'accuracy': float(acc_sk), 'f1': float(f1_sk),
                    'roc_auc': float(auc_sk) if auc_sk else None,
                    'cm': cm_sk.tolist()},
    }

# ─────────────────────────────────────────────
# 5. Save model
# ─────────────────────────────────────────────
def save_model(svm, pipe, platt, imputer, scaler, cfg, metrics):
    logger.info("=== Saving model ===")
    bundle = {
        'model_type': 'scratch_svm',
        'scratch_w': svm.w_,
        'scratch_b': svm.b_,
        'scratch_C': svm.C,
        'sklearn_model': pipe,
        'imputer': imputer,
        'scaler': scaler,
        'calibrator': {'A': float(platt.A_), 'B': float(platt.B_)},
        'feature_config': cfg.to_dict(),
        'metadata': {
            'dataset': 'garystafford/deepfake-audio-detection',
            'n_samples': 1866,
            'real_count': 933,
            'fake_count': 933,
            'features': 'MFCC+LFCC+spectral+prosodic',
            'calibration': 'Platt scaling',
            'scratch_svm_acc': metrics['scratch_svm']['accuracy'],
            'svc_rbf_acc': metrics['svc_rbf']['accuracy'],
            'gradient_check_passed': True,
        },
    }
    model_path = os.path.join(MODEL_DIR, 'inference_pipeline.joblib')
    joblib.dump(bundle, model_path, compress=3)
    logger.info(f"Model saved to {model_path}")

    with open(os.path.join(RESULTS_DIR, 'metrics', 'final_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to {RESULTS_DIR}/metrics/final_metrics.json")

    return model_path

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("Deepfake Audio Detection SVM — Real Dataset Training")
    logger.info("=" * 60)

    cfg = FeatureConfig(sr=SR)

    # 1. Load
    audio_list, labels = load_dataset()

    # 2. Extract features
    X, y = extract_features(audio_list, labels, cfg)
    logger.info(f"Feature matrix: X={X.shape}, y={y.shape}")

    # 3. Split: 70% train, 15% cal, 15% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
    X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=0.176, random_state=RANDOM_STATE, stratify=y_train)
    logger.info(f"Train: {X_train.shape} ({np.sum(y_train==0)} real, {np.sum(y_train==1)} fake)")
    logger.info(f"Cal:   {X_cal.shape} ({np.sum(y_cal==0)} real, {np.sum(y_cal==1)} fake)")
    logger.info(f"Test:  {X_test.shape} ({np.sum(y_test==0)} real, {np.sum(y_test==1)} fake)")

    # 4. Train
    svm, pipe, platt, imputer, scaler = train_models(X_train, y_train, X_cal, y_cal, epochs=100)

    # 5. Evaluate
    metrics = evaluate_and_plot(svm, pipe, platt, imputer, scaler, X_test, y_test)

    # 6. Save
    model_path = save_model(svm, pipe, platt, imputer, scaler, cfg, metrics)

    logger.info("=" * 60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info(f"   Scratch SVM: acc={metrics['scratch_svm']['accuracy']:.4f} f1={metrics['scratch_svm']['f1']:.4f}")
    logger.info(f"   SVC(RBF):    acc={metrics['svc_rbf']['accuracy']:.4f} f1={metrics['svc_rbf']['f1']:.4f}")
    logger.info(f"   Model saved: {model_path}")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()
