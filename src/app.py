"""
app.py
======
EchoVerify AI — Gradio-based interactive deepfake speech detection website.

Branding     : EchoVerify AI
Subtitle     : Deepfake Speech Detection Powered by Support Vector Machines
Theme        : Cybersecurity / AI research laboratory aesthetic
              Dark navy/charcoal background, cyan/teal accents, violet secondary

Launch from Colab with:
    from src.app import build_app
    app = build_app(pipeline)
    app.demo.launch(share=True, debug=True)
"""

import os
import io
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import gradio as gr
    _GRADIO_AVAILABLE = True
except ImportError:
    _GRADIO_AVAILABLE = False
    warnings.warn("Gradio not installed. Run: pip install gradio")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    import librosa
    import librosa.display
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CSS Styling
# ─────────────────────────────────────────────

CUSTOM_CSS = """
/* ═══════════════════════════════════════════
   EchoVerify AI — Custom Gradio CSS
   ═══════════════════════════════════════════ */

:root {
    --bg-primary:    #0d1117;
    --bg-secondary:  #161b22;
    --bg-card:       #1c2128;
    --accent-cyan:   #00d4ff;
    --accent-teal:   #00b4a0;
    --accent-violet: #7c3aed;
    --text-primary:  #e6edf3;
    --text-muted:    #8b949e;
    --border-color:  #30363d;
    --real-color:    #3fb950;
    --fake-color:    #f85149;
    --warn-color:    #d29922;
}

body, .gradio-container {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
}

/* Header */
.echo-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 1.5rem;
}

.echo-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
}

.echo-subtitle {
    color: var(--text-muted);
    font-size: 1rem;
    margin-top: 0.3rem;
}

/* Cards */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

/* Result label */
.result-real {
    color: var(--real-color);
    font-size: 1.8rem;
    font-weight: 700;
    text-align: center;
}

.result-fake {
    color: var(--fake-color);
    font-size: 1.8rem;
    font-weight: 700;
    text-align: center;
}

/* Buttons */
.btn-primary {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-teal)) !important;
    color: #000 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    border: none !important;
}

.btn-secondary {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

/* Tabs */
.tabs > .tab-nav > button {
    color: var(--text-muted) !important;
    border-bottom: 2px solid transparent !important;
}

.tabs > .tab-nav > button.selected {
    color: var(--accent-cyan) !important;
    border-bottom-color: var(--accent-cyan) !important;
}

/* Input area */
.upload-area {
    border: 2px dashed var(--accent-cyan) !important;
    border-radius: 12px !important;
    background: rgba(0, 212, 255, 0.03) !important;
}

/* Warning box */
.warning-box {
    background: rgba(210, 153, 34, 0.1);
    border: 1px solid var(--warn-color);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: var(--warn-color);
    font-size: 0.9rem;
}

/* Footer */
.echo-footer {
    text-align: center;
    padding: 1rem;
    color: var(--text-muted);
    font-size: 0.8rem;
    border-top: 1px solid var(--border-color);
    margin-top: 2rem;
}

/* Override Gradio defaults */
.gr-box, .gr-padded {
    background: var(--bg-card) !important;
    border-color: var(--border-color) !important;
}

label, .gr-block-label {
    color: var(--text-primary) !important;
}

.gr-input, .gr-textarea {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-color) !important;
}
"""

# ─────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────

def _plot_waveform_spectrogram(audio_path: str) -> Optional[object]:
    """
    Generate a waveform + spectrogram figure for the uploaded audio.
    Returns a matplotlib figure or None on failure.
    """
    if not (_MPL_AVAILABLE and _LIBROSA_AVAILABLE):
        return None

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        fig, axes = plt.subplots(2, 1, figsize=(10, 5))
        fig.patch.set_facecolor("#161b22")

        for ax in axes:
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="#8b949e")
            ax.spines[:].set_color("#30363d")

        # Waveform
        t = np.linspace(0, len(y) / sr, len(y))
        axes[0].plot(t, y, color="#00d4ff", linewidth=0.6, alpha=0.9)
        axes[0].set_ylabel("Amplitude", color="#8b949e")
        axes[0].set_xlabel("Time (s)", color="#8b949e")
        axes[0].set_title("Waveform", color="#e6edf3", fontsize=11)
        axes[0].set_xlim(0, t[-1])

        # Mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(
            S_db, x_axis="time", y_axis="mel", sr=sr, fmax=8000,
            ax=axes[1], cmap="magma"
        )
        fig.colorbar(img, ax=axes[1], format="%+2.0f dB", label="dB")
        axes[1].set_title("Mel Spectrogram", color="#e6edf3", fontsize=11)
        axes[1].set_xlabel("Time (s)", color="#8b949e")
        axes[1].set_ylabel("Frequency (Hz)", color="#8b949e")

        plt.tight_layout(pad=1.5)
        return fig
    except Exception as exc:
        logger.warning(f"Visualisation failed: {exc}")
        return None


def _plot_probability_bar(p_real: float, p_fake: float) -> Optional[object]:
    """
    Generate a horizontal probability bar chart.
    """
    if not _MPL_AVAILABLE:
        return None

    try:
        fig, ax = plt.subplots(figsize=(8, 1.6))
        fig.patch.set_facecolor("#161b22")
        ax.set_facecolor("#161b22")

        ax.barh(["FAKE", "REAL"], [p_fake * 100, p_real * 100],
                color=["#f85149", "#3fb950"], height=0.4, edgecolor="none")

        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)", color="#8b949e", fontsize=9)
        ax.tick_params(colors="#e6edf3", labelsize=10)
        ax.spines[:].set_color("#30363d")
        ax.set_title("Classification Probabilities", color="#e6edf3", fontsize=10)

        # Annotate bars
        ax.text(p_fake * 100 + 1, 1, f"{p_fake*100:.1f}%", va="center",
                color="#f85149", fontsize=11, fontweight="bold")
        ax.text(p_real * 100 + 1, 0, f"{p_real*100:.1f}%", va="center",
                color="#3fb950", fontsize=11, fontweight="bold")

        plt.tight_layout()
        return fig
    except Exception as exc:
        logger.warning(f"Probability bar plot failed: {exc}")
        return None


def _plot_segment_timeline(segment_proba: list) -> Optional[object]:
    """
    Plot per-segment fake probability as a timeline.
    """
    if not _MPL_AVAILABLE or not segment_proba:
        return None

    try:
        p_fakes = [p[1] for p in segment_proba]
        fig, ax = plt.subplots(figsize=(10, 2.5))
        fig.patch.set_facecolor("#161b22")
        ax.set_facecolor("#0d1117")

        x = np.arange(len(p_fakes))
        colors = ["#f85149" if p >= 0.5 else "#3fb950" for p in p_fakes]
        ax.bar(x, [p * 100 for p in p_fakes], color=colors, width=0.7, edgecolor="none")
        ax.axhline(50, color="#8b949e", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Segment Index", color="#8b949e")
        ax.set_ylabel("FAKE Probability (%)", color="#8b949e")
        ax.set_title("Per-Segment FAKE Probability", color="#e6edf3", fontsize=10)
        ax.set_ylim(0, 100)
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")

        plt.tight_layout()
        return fig
    except Exception as exc:
        logger.warning(f"Segment timeline plot failed: {exc}")
        return None


# ─────────────────────────────────────────────
# App builder
# ─────────────────────────────────────────────

def build_app(pipeline):
    """
    Build and return the Gradio Blocks application.

    Parameters
    ----------
    pipeline : InferencePipeline
        A loaded and tested inference pipeline.

    Returns
    -------
    gr.Blocks  demo object (call .launch() to start the server)
    """
    if not _GRADIO_AVAILABLE:
        raise RuntimeError(
            "Gradio not installed. Install with: pip install gradio"
        )

    # ── Inference callback ────────────────────────────────────────────
    def analyze_audio(audio_input):
        """
        Called by the Gradio interface when the user clicks Analyze.

        audio_input : path string (from gr.Audio with type="filepath")
        """
        if audio_input is None:
            return (
                "⚠️ No audio provided. Please upload a file or record using the microphone.",
                None, None, None,
                "— | —",
                "Please upload audio first.",
                "",
            )

        from .inference import predict_audio as _predict
        result = _predict(audio_input, pipeline)

        if not result["valid"] or result["label_str"] is None:
            err_msg = result.get("error") or result.get("warning") or "Analysis failed."
            return (
                f"❌ {err_msg}",
                None, None, None,
                "— | —",
                err_msg,
                "",
            )

        label_str = result["label_str"]
        p_real = result["p_real"]
        p_fake = result["p_fake"]
        confidence = result["confidence"]
        warning = result.get("warning", "")

        # Classification label HTML
        icon = "🔴" if label_str == "FAKE" else "🟢"
        label_color = "#f85149" if label_str == "FAKE" else "#3fb950"
        label_html = (
            f"<div style='text-align:center; padding:1rem;'>"
            f"<div style='font-size:3rem;'>{icon}</div>"
            f"<div style='font-size:2rem; font-weight:700; color:{label_color};'>"
            f"{'AI-GENERATED / FAKE SPEECH' if label_str == 'FAKE' else 'REAL HUMAN SPEECH'}"
            f"</div>"
            f"<div style='font-size:0.9rem; color:#8b949e; margin-top:0.5rem;'>"
            f"Confidence: {confidence.upper()}"
            f"</div>"
            f"</div>"
        )

        # Probability summary
        prob_summary = (
            f"🟢 REAL: {p_real*100:.1f}%  |  🔴 FAKE: {p_fake*100:.1f}%"
        )

        # Explanation
        if label_str == "FAKE":
            explanation = (
                "The model found acoustic patterns more consistent with the "
                "AI-generated speech samples observed during training. "
                "This is a model-estimated probability, not a direct measurement "
                "of synthetic content in the recording."
            )
        else:
            explanation = (
                "The model found acoustic patterns more consistent with the "
                "genuine human speech samples observed during training. "
                "This is a model-estimated probability, not a guarantee of authenticity."
            )

        if warning:
            explanation += f"\n\n⚠️ {warning}"

        # Diagnostics
        diag = result.get("diagnostics", {})
        dur = diag.get("duration_s", 0.0)
        sr = diag.get("sr", 16000)
        n_seg = diag.get("n_segments", 1)
        diag_text = (
            f"Duration: {dur:.2f}s | Sample Rate: {sr} Hz | "
            f"Segments processed: {n_seg} | Model: {pipeline.model_name}"
        )

        # Plots
        prob_bar_fig = _plot_probability_bar(p_real, p_fake)
        signal_fig = _plot_waveform_spectrogram(audio_input)
        seg_fig = _plot_segment_timeline(result.get("segment_proba", []))

        return (
            label_html,
            prob_bar_fig,
            signal_fig,
            seg_fig,
            prob_summary,
            explanation,
            diag_text,
        )

    # ── Gradio layout ─────────────────────────────────────────────────
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="EchoVerify AI — Deepfake Speech Detection",
        theme=gr.themes.Base(
            primary_hue="cyan",
            secondary_hue="violet",
            neutral_hue="slate",
        ),
    ) as demo:

        # ─ Header ─────────────────────────────────────────────────────
        gr.HTML("""
        <div class="echo-header">
            <div class="echo-title">🔊 EchoVerify AI</div>
            <div class="echo-subtitle">
                Deepfake Speech Detection Powered by Support Vector Machines
            </div>
            <div style="margin-top:0.5rem; color:#8b949e; font-size:0.8rem;">
                Course: 22MAT220 – Mathematics for Computing III &nbsp;|&nbsp;
                Theme A: Optimization &amp; Machine Learning
            </div>
        </div>
        """)

        # ─ Main layout ────────────────────────────────────────────────
        with gr.Row():
            # Left column: input
            with gr.Column(scale=1):
                gr.Markdown("### 🎤 Audio Input")
                gr.Markdown(
                    "Upload an audio file or record directly using your microphone. "
                    "Supported formats: WAV, MP3, FLAC, OGG, M4A"
                )
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Audio Input",
                    elem_classes=["upload-area"],
                )
                with gr.Row():
                    analyze_btn = gr.Button(
                        "🔍 Analyze Audio",
                        elem_classes=["btn-primary"],
                        variant="primary",
                    )
                    clear_btn = gr.ClearButton(
                        [audio_input],
                        value="🗑️ Clear",
                        elem_classes=["btn-secondary"],
                    )

                gr.Markdown(
                    "**Note:** The analysis uses the trained SVM model loaded at startup. "
                    "No retraining occurs on upload."
                )

            # Right column: results
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Analysis Result")
                label_output = gr.HTML(
                    value="<div style='text-align:center; color:#8b949e; padding:2rem;'>"
                    "Upload audio and click Analyze to see results.</div>"
                )
                prob_summary_output = gr.Textbox(
                    label="Probability Summary",
                    interactive=False,
                    value="—",
                )
                prob_bar_output = gr.Plot(label="Classification Probabilities")

        # ─ Detail tabs ────────────────────────────────────────────────
        with gr.Tabs():
            with gr.TabItem("📈 Signal Analysis"):
                signal_plot = gr.Plot(label="Waveform & Spectrogram")
                diag_output = gr.Textbox(label="Signal Diagnostics", interactive=False)

            with gr.TabItem("🕐 Segment Timeline"):
                seg_plot = gr.Plot(label="Per-Segment FAKE Probability")
                gr.Markdown(
                    "Each bar represents one audio segment (~3 seconds). "
                    "Segments above 50% are classified as FAKE (red), below as REAL (green). "
                    "The final prediction is the mean probability across all segments."
                )

            with gr.TabItem("💬 Explanation"):
                explanation_output = gr.Textbox(
                    label="Model Explanation",
                    interactive=False,
                    lines=6,
                )
                gr.Markdown("""
                **Interpreting the result:**

                - The probabilities are model-estimated values based on acoustic features (MFCC, LFCC, spectral features).
                - They indicate how consistent the audio is with the training distribution of REAL vs FAKE speech.
                - A "FAKE" result means the SVM decision function lies on the positive-class side of the learned hyperplane.
                - Probabilities are calibrated using Platt scaling to be meaningful estimates, not raw scores.
                - The model cannot guarantee detection of all deepfake techniques, especially those not seen during training.
                """)

            with gr.TabItem("ℹ️ Model Info"):
                metadata = pipeline.metadata if hasattr(pipeline, "metadata") else {}
                info_md = f"""
                **Model:** {getattr(pipeline, 'model_name', 'SVM')}

                **Dataset:** {metadata.get('dataset', 'ASVspoof 2019 LA')}

                **Features:** MFCC + LFCC + Spectral + Prosodic

                **Calibration:** Platt Scaling (sigmoid)

                **Important Limitations:**
                - The model was trained on specific deepfake generators; novel generators may evade detection.
                - Performance may degrade on speech recorded in very different acoustic conditions.
                - Short recordings (<0.5s) or non-speech audio are rejected.
                - Probabilities are model estimates, not direct measurements of synthetic content.
                - This system should not be used as the sole evidence in any consequential decision.

                **Evaluation:**
                - Internal test set accuracy: {metadata.get('internal_test_acc', 'see notebook')}
                - External test set accuracy: {metadata.get('external_test_acc', 'see notebook')}
                """
                gr.Markdown(info_md)

            with gr.TabItem("📚 About"):
                gr.Markdown("""
                ## What is Deepfake Audio?

                Deepfake audio refers to speech that has been artificially synthesised or voice-converted
                using machine learning techniques, including:
                - **Text-to-Speech (TTS)** systems (e.g., neural vocoders)
                - **Voice Conversion (VC)** systems that transfer one speaker's voice to another
                - **Speech synthesis** with naturalness indistinguishable from human speech

                ## How Does the SVM Detector Work?

                1. **Feature Extraction:** The audio is converted to acoustic features (MFCC, LFCC,
                   spectral features) that capture properties of the speech signal.
                2. **SVM Classification:** A Support Vector Machine finds the optimal hyperplane
                   separating REAL and FAKE feature distributions in the feature space.
                3. **Probability Calibration:** Raw SVM decision scores are mapped to probabilities
                   using Platt scaling (a sigmoid function fitted on a calibration set).
                4. **Segment Aggregation:** For longer audio, predictions are made per 3-second
                   segment and averaged to produce a final probability.

                ## Why Can Predictions Be Uncertain?

                - The SVM was trained on specific deepfake generators; novel generators produce
                  different spectral artefacts not seen during training.
                - Recording conditions, codec compression, and background noise affect feature quality.
                - Very short recordings provide insufficient evidence for reliable classification.

                ## Mathematical Foundation

                The SVM solves:

                **min_{w,b}** ½‖w‖² + (C/n) Σᵢ max(0, 1 - yᵢ(w·xᵢ + b))

                where the hinge loss penalises margin violations and C controls the trade-off
                between margin width and training error.

                ## Project Information

                - **Course:** 22MAT220 – Mathematics for Computing III
                - **Theme:** Theme A — Optimization & Machine Learning
                - **Algorithm:** Support Vector Machine (from scratch + sklearn)
                """)

        # ─ Analyze button callback ─────────────────────────────────────
        analyze_btn.click(
            fn=analyze_audio,
            inputs=[audio_input],
            outputs=[
                label_output,
                prob_bar_output,
                signal_plot,
                seg_plot,
                prob_summary_output,
                explanation_output,
                diag_output,
            ],
        )

        # ─ Footer ─────────────────────────────────────────────────────
        gr.HTML("""
        <div class="echo-footer">
            EchoVerify AI &nbsp;|&nbsp;
            22MAT220 Mathematics for Computing III &nbsp;|&nbsp;
            Theme A: Optimization &amp; Machine Learning &nbsp;|&nbsp;
            SVM-based Deepfake Speech Detection &nbsp;|&nbsp;
            <em>Predictions are model estimates. Not for use as sole forensic evidence.</em>
        </div>
        """)

    return demo
