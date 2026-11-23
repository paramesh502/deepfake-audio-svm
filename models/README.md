# Models Directory

This directory contains saved model artifacts generated during training.

## Expected Files After Training

| File | Description |
|------|-------------|
| `inference_pipeline.joblib` | Complete inference pipeline (model + scaler + calibrator + config) |
| `scratch_svm_best.joblib` | Best from-scratch SVM + its scaler/imputer |
| `sklearn_linear_svc.joblib` | Fitted scikit-learn LinearSVC pipeline |
| `sklearn_svc_linear.joblib` | Fitted scikit-learn SVC(linear) pipeline |
| `sklearn_svc_rbf.joblib` | Fitted scikit-learn SVC(RBF) pipeline |
| `calibrator_platt.joblib` | Platt calibration parameters for scratch SVM |
| `model_metadata.json` | Version, dataset, feature config, evaluation summary |
| `feature_config.json` | Feature extraction configuration (required for inference) |
| `class_mapping.json` | `{"0": "REAL", "1": "FAKE"}` |

## Loading the Inference Pipeline

```python
from src.inference import InferencePipeline

pipeline = InferencePipeline.load("models/inference_pipeline.joblib")

# Predict on a new audio file
from src.inference import predict_audio
result = predict_audio("path/to/audio.wav", pipeline)
print(f"Label: {result['label_str']}")
print(f"P(REAL): {result['p_real']*100:.1f}%")
print(f"P(FAKE): {result['p_fake']*100:.1f}%")
```

## Model Architecture

The deployed model is a **scikit-learn SVC with RBF kernel** (or the best-performing
model selected during validation), wrapped in a Pipeline:

```
SimpleImputer(strategy='mean')
    → StandardScaler()
    → SVC(kernel='rbf', probability=True, C=best_C, gamma=best_gamma)
    → CalibratedClassifierCV (Platt scaling)
```

## Important Notes

- Model files are **not** committed to version control (too large).
- Run the Colab notebook (Cell 32) to generate them.
- For Google Colab, mount your Drive and configure `configs/experiment_config.yaml`
  with the correct `drive_project_folder` path.
- Never load model files from untrusted sources; joblib files can execute arbitrary code.

## Reload Verification

The notebook (Cell 33) verifies that:
1. The saved model is loadable
2. Predictions on a test sample match within numerical tolerance
3. Probabilities sum to 1.0 within floating-point precision
