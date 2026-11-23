# Faculty Demonstration Instructions

## Pre-Demo Checklist
1. Ensure Google Colab is accessible (https://colab.research.google.com)
2. Have ASVspoof 2019 LA dataset uploaded to Google Drive (for full mode)
   OR use Quick Demo mode (500 samples, no full dataset needed)
3. Prepare 1-2 test audio files (one real, one fake) not in the dataset

## Demo Flow

### Step 1: Open the Notebook
- Upload `Deepfake_Audio_Detection_SVM_Colab.ipynb` to Colab
- Set runtime type: Runtime → Change runtime type → Python 3 + GPU (optional)

### Step 2: Run Sequentially
- Cells 03-04: Install dependencies + import libraries
- Cell 05: Mount Google Drive
- Cells 06-09: Dataset configuration and exploration
- Cells 10-14: Preprocessing and feature extraction
- Cells 15-16: Build feature matrix and splits
- Cells 18-21: Mathematical SVM + convergence analysis
- Cells 22-26: sklearn baselines + hyperparameter tuning + comparison
- Cell 27: Probability calibration
- Cell 28: Internal evaluation (confusion matrix, ROC, PR curves)
- Cell 32: Save model
- Cell 33: Reload model (consistency check)
- Cells 36-38: Build and launch Gradio website

### Step 3: Live Website Demo
1. Run Cell 37 to launch the Gradio app
2. Click the shareable URL (https://xxxx.gradio.live)
3. Upload a test audio file (WAV/MP3/FLAC)
4. Click "Analyze Audio"
5. View:
   - REAL/FAKE label with probabilities
   - Waveform and spectrogram
   - Per-segment timeline
6. Try recording via microphone (if browser allows)

## Viva Q&A Topics to Cover
- Why SVM? (mathematical foundations, interpretability, course alignment)
- Hinge loss derivation
- Subgradient computation
- Support vectors and margins
- Feature scaling importance
- MFCC vs LFCC differences
- Platt scaling purpose
- Generalization gap explanation
- Why novel deepfakes may evade detection

See `documentation/viva_questions.md` for detailed Q&A.
