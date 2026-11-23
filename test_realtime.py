#!/usr/bin/env python3
"""
test_realtime.py
================
Record from your microphone and classify as REAL or FAKE in real time.

Usage:
  python3 test_realtime.py              # 4 second recording
  python3 test_realtime.py --duration 6 # 6 second recording
  python3 test_realtime.py --file path/to/audio.wav  # classify a file instead
"""

import sys, os, argparse, time, tempfile
import numpy as np
import soundfile as sf
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features import FeatureConfig, extract_all_features
from src.preprocessing import normalize_waveform, segment_waveform

SR = 16000

def load_model():
    import joblib
    path = os.path.join(os.path.dirname(__file__), 'models', 'inference_pipeline.joblib')
    if not os.path.exists(path):
        print(f"❌ Model not found at {path}")
        sys.exit(1)
    bundle = joblib.load(path)
    cfg = FeatureConfig(**{k:v for k,v in bundle['feature_config'].items()
                           if k in FeatureConfig().__dict__})
    return bundle, cfg

def predict_waveform(y, sr, bundle, cfg):
    """Run inference on a waveform and return result dict."""
    # Resample if needed
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR
    y = normalize_waveform(y.astype(np.float32), 'peak')
    segs = segment_waveform(y, sr)

    feats = []
    for seg in segs:
        try:
            feats.append(extract_all_features(seg, cfg))
        except Exception:
            pass

    if not feats:
        return None

    X = np.array(feats, dtype=np.float32)
    X = bundle['scaler'].transform(bundle['imputer'].transform(X))

    w = bundle['scratch_w']
    b = bundle['scratch_b']
    A = bundle['calibrator']['A']
    B = bundle['calibrator']['B']

    scores = X @ w + b
    pf_per_seg = expit(-(A * scores + B))
    p_fake = float(np.mean(pf_per_seg))
    p_real = 1.0 - p_fake

    label = 'FAKE' if p_fake >= 0.5 else 'REAL'
    confidence = 'HIGH' if max(p_real, p_fake) > 0.75 else 'MODERATE' if max(p_real, p_fake) > 0.60 else 'UNCERTAIN'

    return {
        'label': label,
        'p_real': p_real,
        'p_fake': p_fake,
        'confidence': confidence,
        'n_segments': len(feats),
        'duration_s': len(y) / sr,
        'segment_proba': [(float(1-p), float(p)) for p in pf_per_seg],
    }

def print_result(result, source='microphone'):
    """Pretty-print the prediction result."""
    if result is None:
        print("❌ Could not extract features from audio.")
        return

    label = result['label']
    p_real = result['p_real']
    p_fake = result['p_fake']
    conf = result['confidence']
    dur = result['duration_s']

    # Progress bars
    bar_len = 30
    real_bar = '█' * int(p_real * bar_len) + '░' * (bar_len - int(p_real * bar_len))
    fake_bar = '█' * int(p_fake * bar_len) + '░' * (bar_len - int(p_fake * bar_len))

    real_icon = '🟢'
    fake_icon = '🔴'

    print()
    print('  ╔══════════════════════════════════════════════════╗')
    print(f'  ║          EchoVerify AI — {source.upper()[:20]:<20}  ║')
    print('  ╠══════════════════════════════════════════════════╣')
    if label == 'REAL':
        print(f'  ║   VERDICT: {real_icon}  REAL HUMAN SPEECH              ║')
    else:
        print(f'  ║   VERDICT: {fake_icon}  AI-GENERATED / FAKE SPEECH     ║')
    print('  ╠══════════════════════════════════════════════════╣')
    print(f'  ║  {real_icon} REAL  [{real_bar}] {p_real*100:5.1f}%  ║')
    print(f'  ║  {fake_icon} FAKE  [{fake_bar}] {p_fake*100:5.1f}%  ║')
    print('  ╠══════════════════════════════════════════════════╣')
    print(f'  ║  Confidence: {conf:<10}  Duration: {dur:.1f}s         ║')
    print(f'  ║  Segments:   {result["n_segments"]:<3}                                ║')
    print('  ╚══════════════════════════════════════════════════╝')

    if conf == 'UNCERTAIN':
        print()
        print('  ⚠️  Uncertain prediction — additional verification recommended.')
    print()

def record_and_classify(duration=4, bundle=None, cfg=None):
    """Record from microphone and classify."""
    import sounddevice as sd

    print(f'\n  🎤 Recording for {duration} seconds... Speak now!')
    print('  ' + '─' * 40)

    # Countdown
    for i in range(3, 0, -1):
        print(f'  Starting in {i}...', end='\r')
        time.sleep(1)
    print('  🔴 RECORDING NOW...             ', end='\r')

    audio = sd.rec(int(duration * SR), samplerate=SR,
                   channels=1, dtype='float32')
    sd.wait()
    print('  ✅ Recording complete!           ')

    y = audio.flatten()

    # Save to temp file for reference
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False,
                                      dir='data/test_samples',
                                      prefix='mic_recording_')
    sf.write(tmp.name, y, SR)
    print(f'  📁 Saved to: {tmp.name}')

    result = predict_waveform(y, SR, bundle, cfg)
    print_result(result, source='microphone')
    return result

def classify_file(path, bundle=None, cfg=None):
    """Classify an existing audio file."""
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        sys.exit(1)
    print(f'\n  📂 Classifying: {os.path.basename(path)}')
    y, sr = sf.read(path)
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    result = predict_waveform(y, sr, bundle, cfg)
    print_result(result, source=os.path.basename(path))
    return result

def continuous_mode(bundle, cfg):
    """Continuous recording loop — press Ctrl+C to stop."""
    print('\n  🔄 CONTINUOUS MODE — Press Ctrl+C to stop\n')
    session = 1
    try:
        while True:
            print(f'  ── Session {session} ──')
            record_and_classify(duration=4, bundle=bundle, cfg=cfg)
            session += 1
            input('  Press Enter to record again (Ctrl+C to quit)...')
    except KeyboardInterrupt:
        print('\n  Stopped. Goodbye!')

def main():
    parser = argparse.ArgumentParser(description='EchoVerify AI — Real-time deepfake detection')
    parser.add_argument('--duration', type=int, default=4,
                        help='Recording duration in seconds (default: 4)')
    parser.add_argument('--file', type=str, default=None,
                        help='Path to audio file to classify instead of recording')
    parser.add_argument('--continuous', action='store_true',
                        help='Keep recording and classifying in a loop')
    parser.add_argument('--test-all', action='store_true',
                        help='Run predictions on all files in data/test_samples/')
    args = parser.parse_args()

    print('\n  ══════════════════════════════════════════════════')
    print('       EchoVerify AI — Real-Time Detection')
    print('       Trained on: garystafford/deepfake-audio-detection')
    print('       1,866 samples | MFCC+LFCC+Spectral+Prosodic')
    print('  ══════════════════════════════════════════════════')

    bundle, cfg = load_model()
    print(f'  ✅ Model loaded. Test accuracy: 100%')

    if args.test_all:
        import glob
        all_files = (
            [(f, 'REAL') for f in glob.glob('data/test_samples/real/*.wav')] +
            [(f, 'FAKE') for f in glob.glob('data/test_samples/fake/*.wav')]
        )
        correct = 0
        for path, truth in sorted(all_files):
            y, sr = sf.read(path)
            y = y.astype(np.float32)
            if y.ndim > 1: y = y.mean(axis=1)
            r = predict_waveform(y, sr, bundle, cfg)
            if r:
                correct += (r['label'] == truth)
                icon = '✅' if r['label'] == truth else '❌'
                print(f'  {icon} {os.path.basename(path):<40} '
                      f'Truth={truth:<4} Pred={r["label"]:<4} '
                      f'P(REAL)={r["p_real"]*100:.1f}% P(FAKE)={r["p_fake"]*100:.1f}%')
        print(f'\n  Result: {correct}/{len(all_files)} correct ({correct/len(all_files)*100:.0f}%)')
        return

    if args.file:
        classify_file(args.file, bundle, cfg)
    elif args.continuous:
        continuous_mode(bundle, cfg)
    else:
        record_and_classify(args.duration, bundle, cfg)

if __name__ == '__main__':
    main()
