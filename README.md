# Deepfake Audio Detection Using SVMs
### 22MAT220 – Mathematics for Computing III | Theme A: Optimization & Machine Learning

> **EchoVerify AI** — Binary detection of real vs. AI-generated speech using Support Vector Machines with calibrated probabilistic output.

---

## Project Overview

This project implements a complete, academically rigorous deepfake audio detection system:

- **SVM from scratch** in NumPy with subgradient descent and convergence analysis
- **Comparison** with scikit-learn LinearSVC, SVC(RBF), Logistic Regression, Random Forest
- **Acoustic features**: MFCC, LFCC, spectral features, prosodic features
- **Probability calibration** using Platt scaling with ECE evaluation
- **External generalization** evaluation on unseen datasets
- **Interactive Gradio web app** (EchoVerify AI) launchable from Google Colab

---

## Quick Start (Google Colab)

```python
# 1. Mount Drive and clone/upload the project
from google.colab import drive
drive.mount('/content/drive')

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Open and run the notebook
# notebooks/Deepfake_Audio_Detection_SVM_Colab.ipynb
```

---

## Repository Structure

```
deepfake-audio-svm/
├── notebooks/
│   └── Deepfake_Audio_Detection_SVM_Colab.ipynb  # Main Colab notebook
├── src/
│   ├── preprocessing.py       # Audio loading, validation, segmentation
│   ├── features.py            # MFCC, LFCC, spectral, prosodic features
│   ├── svm_from_scratch.py    # NumPy SVM with subgradient descent
│   ├── model_training.py      # Training orchestration for all models
│   ├── calibration.py         # Platt scaling, ECE, reliability diagrams
│   ├── evaluation.py          # Metrics: accuracy, F1, AUC, EER, etc.
│   ├── inference.py           # Standalone inference pipeline
│   └── app.py                 # Gradio EchoVerify AI web application
├── configs/
│   └── experiment_config.yaml # All hyperparameters and paths
├── models/
│   └── README.md              # Model artifact documentation
├── results/
│   ├── figures/               # Generated plots (PNG/PDF)
│   ├── tables/                # CSV results tables
│   └── metrics/               # JSON metrics per experiment
├── tests/
│   ├── test_preprocessing.py  # Preprocessing unit tests
│   ├── test_features.py       # Feature extraction unit tests
│   ├── test_svm.py            # SVM correctness tests (gradient check)
│   └── test_inference.py      # End-to-end inference tests
├── report/
│   └── research_report.tex    # IEEE-format research paper
├── documentation/
│   ├── mathematical_derivations.md
│   ├── dataset_instructions.md
│   ├── viva_questions.md
│   ├── limitations.md
│   └── demo_instructions.md
├── requirements.txt
└── README.md
```

---

## Dataset Setup

This project uses the **ASVspoof 2019 Logical Access** dataset as the primary training source.

### Download Instructions
1. Register at [https://www.asvspoof.org/index2019.html](https://www.asvspoof.org/index2019.html)
2. Download the LA partition (~2.6 GB)
3. Upload to Google Drive: `MyDrive/22MAT220_Deepfake_SVM/data/`
4. For external evaluation: [In-the-Wild dataset](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild)

For Quick Demo Mode (500 samples, no full download), set in `configs/experiment_config.yaml`:
```yaml
dataset:
  quick_demo:
    max_train_samples: 500
```

---

## Running Tests

```bash
cd deepfake-audio-svm
python -m pytest tests/ -v
```

Expected: All tests in `test_svm.py` and `test_preprocessing.py` pass without audio libraries.
Tests in `test_features.py` and `test_inference.py` require librosa + soundfile.

---

## SVM Implementation (Mathematical Core)

The from-scratch SVM in `src/svm_from_scratch.py` implements:

**Primal objective:**
```
min_{w,b}  (1/2)||w||²  +  (C/n) Σᵢ max(0, 1 - yᵢ(w·xᵢ + b))
```

**Subgradient of w:**
```
∂J/∂w = w                              if yᵢ(w·xᵢ+b) ≥ 1
∂J/∂w = w - (C/n) yᵢ xᵢ              if yᵢ(w·xᵢ+b) < 1
```

**Pegasos learning rate:** `ηₜ = 1/(λ·t)` where `λ = 1/C`

---

## Gradio Web Application

Launch the interactive website from the final Colab cell:

```python
from src.app import build_app
from src.inference import InferencePipeline

pipeline = InferencePipeline.load("models/inference_pipeline.joblib")
demo = build_app(pipeline)
demo.launch(share=True, debug=True)
```

The app accepts:
- Audio file upload (WAV, MP3, FLAC, OGG, M4A)
- Microphone recording
- Returns: REAL/FAKE label + calibrated probabilities + waveform/spectrogram

---

## Limitations

- The system was trained on ASVspoof 2019 attack types. Novel deepfake generators may evade detection.
- External performance is typically lower than internal performance (domain gap).
- Colab hosting requires an active runtime session.
- **Not for forensic use.** Probabilities are model estimates, not absolute truth.

See `documentation/limitations.md` for the full list.

---

## References

1. Wang et al. (2020). ASVspoof 2019. *Computer Speech & Language*.
2. Sahidullah et al. (2015). A Comparison of Features for Synthetic Speech Detection. *Interspeech*.
3. Platt (1999). Probabilistic Outputs for SVMs. *Advances in Large Margin Classifiers*.
4. Shalev-Shwartz et al. (2007). Pegasos. *ICML*.
5. Jung et al. (2022). AASIST. *ICASSP*.
6. Müller et al. (2022). Does Audio Deepfake Detection Generalize? *Interspeech*.

---

## Citation

If you use this code for academic work, cite as:
```
[Author Names]. Deepfake Audio Detection Using SVMs with Calibrated Probabilistic Inference.
22MAT220 Course Project, Amrita School of Engineering, 2026.
```
