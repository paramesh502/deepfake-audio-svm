# Methodology

## 1. System Overview

The EchoVerify AI system is a binary classifier that distinguishes real human
speech from AI-generated speech using Support Vector Machines trained on
handcrafted acoustic features.

## 2. Data Pipeline

### 2.1 Audio Loading
- All audio is decoded to 16 kHz mono using librosa.
- Files are validated for format (WAV, MP3, FLAC, OGG, M4A) and size (<200 MB).
- Stereo audio is converted to mono by averaging channels.

### 2.2 Preprocessing
1. **Duration check**: Reject clips < 0.5s (insufficient speech).
2. **Clipping detection**: Warn if >1% of samples are at ±1.0.
3. **Speech activity**: Estimate voiced frame ratio; warn if < 10%.
4. **Normalization**: Peak normalization to [-1, 1].
5. **Segmentation**: 3-second windows with 1.5-second hop (50% overlap).

### 2.3 Feature Extraction
Each segment produces a fixed-length feature vector:

| Feature Group | Components | Dimension |
|---------------|-----------|-----------|
| MFCC | 20 coefficients + Δ + ΔΔ × 6 stats | 360 |
| LFCC | 20 coefficients + Δ + ΔΔ × 6 stats | 360 |
| Spectral | Centroid, BW, Rolloff, Flatness, Contrast(7), ZCR × 6 stats | 84 |
| Prosodic | RMS, ΔRMS, F0 × 6 stats + voiced ratio | 19 |
| **Total** | | **~823** |

Statistics: mean, std, median, 25th percentile, 75th percentile, max.

### 2.4 Feature Scaling
StandardScaler (mean=0, std=1) is fitted on training data only.
The same scaler is applied to dev, eval, and inference-time features.

## 3. Models

### 3.1 From-Scratch Linear SVM
- Objective: (1/2)||w||^2 + (C/n) Σ hinge loss
- Optimization: Mini-batch subgradient descent
- Learning rate: Pegasos schedule η_t = 1/(λ·t)
- Batch size: 64
- Max epochs: 100 (with early stopping)
- Labels: {0,1} → converted to {-1,+1} internally

### 3.2 Scikit-Learn Models
All wrapped in Pipeline: SimpleImputer → StandardScaler → Classifier

| Model | Classifier | Hyperparameters |
|-------|-----------|-----------------|
| LinearSVC | LinearSVC(dual=False) | C ∈ {0.01, 0.1, 1, 10, 100} |
| SVC(linear) | SVC(kernel='linear') | C ∈ {0.01, 0.1, 1, 10, 100} |
| SVC(RBF) | SVC(kernel='rbf') | C × gamma grid |
| LogisticReg | LogisticRegression | C ∈ {0.01, 0.1, 1, 10, 100} |
| RandomForest | RandomForestClassifier(200 trees) | — |

### 3.3 Hyperparameter Tuning
- 3-fold stratified cross-validation on training partition
- Scoring: F1 score (positive class = FAKE)
- Best model selected by validation F1, NOT test set performance

## 4. Probability Calibration

Raw SVM decision scores are mapped to probabilities using Platt scaling:

    P(y=1|f) = 1 / (1 + exp(A·f + B))

Parameters A, B are fitted on a separate calibration set (15% of training data)
by minimizing the cross-entropy loss with label smoothing.

Calibration quality is evaluated using:
- Brier score (lower is better)
- Log loss (lower is better)
- Expected Calibration Error (ECE)
- Reliability diagrams

## 5. Inference Pipeline

For each uploaded audio file:
1. Load and validate
2. Resample to 16 kHz mono
3. Normalize
4. Segment into 3-second windows
5. Extract features per segment
6. Apply saved scaler
7. Get per-segment calibrated probabilities
8. Aggregate: mean probability across segments
9. Report final REAL/FAKE label + P(REAL) + P(FAKE)

## 6. Evaluation Protocol

### 6.1 Internal Test (Held-Out)
ASVspoof 2019 LA evaluation partition (not used in training or tuning).
Metrics: Accuracy, Balanced Accuracy, Precision, Recall, F1, Specificity,
ROC-AUC, PR-AUC, EER, Brier Score, Confusion Matrix.

### 6.2 External Test
In-the-Wild dataset or user-provided audio.
Same metrics + generalization gap analysis.

### 6.3 Robustness
Noise injection, compression, volume variation, resampling.

## 7. Implementation Notes

- All code is reproducible with RANDOM_STATE = 42.
- Features are cached to disk to avoid recomputation.
- The Gradio app loads the model once at startup (no retraining per inference).
- The from-scratch SVM is independently demonstrable (see cells 19-21).
