#!/usr/bin/env python3
"""
run_app.py
===========
Launch the EchoVerify AI Gradio web app using a previously trained model.

Usage:
  python run_app.py
  python run_app.py --model ./models/inference_pipeline.joblib
  python run_app.py --share   # create a public shareable link

This skips training entirely and just serves the web interface.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Launch EchoVerify AI web app")
    parser.add_argument(
        "--model", default="./models/inference_pipeline.joblib",
        help="Path to the saved model bundle (default: ./models/inference_pipeline.joblib)",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public shareable Gradio link",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to run the app on (default: 7860)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("   Train a model first with: python run_local.py --data-dir ./data")
        sys.exit(1)

    import joblib
    from src.features import FeatureConfig
    from src.inference import InferencePipeline
    from src.calibration import PlattCalibrator
    from src.svm_from_scratch import LinearSVMScratch
    from src.app import build_app

    print(f"Loading model from {args.model}...")
    bundle = joblib.load(args.model)

    # Build FeatureConfig from saved config
    cfg_kwargs = {k: v for k, v in bundle["feature_config"].items()
                  if k in FeatureConfig().__dict__}
    cfg = FeatureConfig(**cfg_kwargs)

    # Reconstruct calibrator
    calibrator = None
    if bundle.get("calibrator"):
        calibrator = PlattCalibrator.from_dict(bundle["calibrator"])

    # Reconstruct model
    if bundle["model_type"] == "scratch_svm":
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

    print("Building EchoVerify AI interface...")
    demo = build_app(pipeline)
    print(f"\n🚀 Launching EchoVerify AI on http://localhost:{args.port}")
    print("   Press Ctrl+C to stop.\n")
    demo.launch(
        share=args.share,
        debug=True,
        server_port=args.port,
    )


if __name__ == "__main__":
    main()
