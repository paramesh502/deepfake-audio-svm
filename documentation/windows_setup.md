# Windows Setup Guide

This project runs on Windows 10/11 with no code changes needed.
All paths use `os.path.join()` so they work cross-platform automatically.

## Prerequisites

Install Python 3.10 or 3.11 from https://www.python.org/downloads/
During installation check: ✅ Add Python to PATH

## Step-by-Step Setup

### 1. Open PowerShell or Command Prompt
```powershell
# Check Python is installed
python --version
# Should show: Python 3.10.x or 3.11.x
```

### 2. Clone the repository
```powershell
git clone https://github.com/paramesh502/deepfake-audio-svm.git
cd deepfake-audio-svm
```

### 3. Create a virtual environment (recommended)
```powershell
python -m venv venv
venv\Scripts\activate
```

### 4. Install dependencies
```powershell
pip install -r requirements.txt
```

> ⚠️ If `soundfile` fails on Windows, install it with:
> ```powershell
> pip install soundfile
> ```
> soundfile on Windows uses libsndfile which is bundled automatically.

> ⚠️ If `librosa` has issues, make sure you have Microsoft C++ Build Tools:
> Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### 5. Download the dataset
Download both parquet files from:
https://huggingface.co/datasets/garystafford/deepfake-audio-detection/tree/main/data

Place them in:
```
deepfake-audio-svm\
  data\
    train-00000-of-00002.parquet
    train-00001-of-00002.parquet
```

### 6. Train the model
```powershell
python train_real.py
```
This takes about 5-10 minutes (feature extraction + SVM training).
Features are cached after first run — subsequent runs take < 30 seconds.

### 7. Launch the web app
```powershell
python run_app.py --port 7860
```
Open http://localhost:7860 in your browser.

### 8. Real-time microphone test
```powershell
python test_realtime.py
```
Requires a working microphone. Uses Windows audio API automatically via sounddevice.

## Windows-Specific Notes

### MP3 support
librosa uses `audioread` for MP3 on Windows. If MP3 files fail, install ffmpeg:
1. Download ffmpeg from https://ffmpeg.org/download.html
2. Add ffmpeg/bin to your PATH

Or convert MP3 to WAV first:
```powershell
ffmpeg -i input.mp3 output.wav
```

### Microphone permissions
Windows may block microphone access. If `test_realtime.py` fails:
- Settings → Privacy & Security → Microphone → Allow apps to access microphone ✅

### Firewall for Gradio
When running `run_app.py`, Windows Firewall may ask to allow Python.
Click **Allow** to enable the web interface at http://localhost:7860.

### Path lengths
Windows has a 260-character path limit by default. If you see path errors:
- Enable long paths: Settings → System → For Developers → Long Paths ✅
- Or place the project in a short path like `C:\deepfake-svm\`

## Running Tests on Windows
```powershell
python -m pytest tests\test_svm.py tests\test_preprocessing.py -v
```

## Verified Working On
- Windows 11 with Python 3.11
- Windows 10 with Python 3.10
- All paths use os.path.join() — no hardcoded forward slashes

## If Something Fails

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: librosa` | `pip install librosa` |
| `OSError: sndfile library not found` | `pip install soundfile --force-reinstall` |
| `PortAudioError` (sounddevice) | Install PortAudio: `pip install sounddevice` |
| `Microsoft Visual C++ required` | Install VS Build Tools (link above) |
| Gradio page not loading | Check firewall, try `--port 8080` |
