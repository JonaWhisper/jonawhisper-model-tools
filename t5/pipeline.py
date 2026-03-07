#!/usr/bin/env python3
"""Full pipeline: convert T5 models from source (PyTorch) → ONNX FP32 → INT8.

Downloads the original PyTorch models, converts to ONNX via optimum-cli,
quantizes to INT8, and optionally uploads everything to HuggingFace.

Requires: pip install optimum[exporters] onnxruntime huggingface-hub

Usage:
    # Full pipeline for all models
    python t5/pipeline.py

    # Single model
    python t5/pipeline.py --model t5-spell-fr

    # Full pipeline + upload to HuggingFace
    python t5/pipeline.py --upload

    # Force re-run all steps
    python t5/pipeline.py --force
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
MODELS_JSON = SCRIPT_DIR / "models.json"
WORK_DIR = SCRIPT_DIR / "work"

UPLOAD_FILES = [
    "encoder_model.onnx",
    "decoder_model.onnx",
    "encoder_model_int8.onnx",
    "decoder_model_int8.onnx",
    "config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
]


def load_models():
    with open(MODELS_JSON) as f:
        return json.load(f)


def step1_convert(source: str, onnx_dir: Path, force: bool = False):
    """Step 1: PyTorch → ONNX FP32 via optimum-cli."""
    encoder = onnx_dir / "encoder_model.onnx"
    decoder = onnx_dir / "decoder_model.onnx"

    if encoder.exists() and decoder.exists() and not force:
        print(f"  [convert] skip: ONNX already exists")
        return

    print(f"  [convert] {source} → ONNX FP32")

    if onnx_dir.exists():
        shutil.rmtree(onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "optimum-cli", "export", "onnx",
            "--model", source,
            "--task", "text2text-generation",
            str(onnx_dir),
        ],
        check=True,
    )

    for f in sorted(onnx_dir.glob("*.onnx")):
        print(f"    {f.name}: {f.stat().st_size / 1024 / 1024:.1f} MB")


def step2_quantize(onnx_dir: Path, force: bool = False):
    """Step 2: ONNX FP32 → INT8 dynamic quantization."""
    for name in ["encoder_model", "decoder_model"]:
        fp32 = onnx_dir / f"{name}.onnx"
        int8 = onnx_dir / f"{name}_int8.onnx"

        if not fp32.exists():
            print(f"  [quantize] skip: {fp32.name} not found")
            continue

        if int8.exists() and not force:
            print(f"  [quantize] skip: {int8.name} already exists")
            continue

        print(f"  [quantize] {fp32.name} → {int8.name}")
        print(f"    input:  {fp32.stat().st_size / 1024 / 1024:.1f} MB")

        quantize_dynamic(
            str(fp32),
            str(int8),
            weight_type=QuantType.QInt8,
        )

        ratio = int8.stat().st_size / fp32.stat().st_size
        print(f"    output: {int8.stat().st_size / 1024 / 1024:.1f} MB ({ratio:.0%})")


def step3_upload(repo: str, onnx_dir: Path):
    """Step 3: Upload all files to HuggingFace."""
    for name in UPLOAD_FILES:
        f = onnx_dir / name
        if f.exists():
            print(f"  [upload] {name} → {repo}")
            subprocess.run(
                ["huggingface-cli", "upload", repo, str(f), name],
                check=True,
            )


def process_model(model: dict, upload: bool = False, force: bool = False):
    model_id = model["id"]
    source = model["source"]
    repo = model["repo"]

    print(f"\n{'=' * 60}")
    print(f"Model:  {model['label']} ({model_id})")
    print(f"Source: {source}")
    print(f"Target: {repo}")
    print(f"{'=' * 60}")

    onnx_dir = WORK_DIR / model_id

    # Step 1: Convert
    step1_convert(source, onnx_dir, force=force)

    # Step 2: Quantize
    step2_quantize(onnx_dir, force=force)

    # Step 3: Upload
    if upload:
        step3_upload(repo, onnx_dir)


def main():
    parser = argparse.ArgumentParser(description="Full pipeline: PyTorch → ONNX FP32 → INT8")
    parser.add_argument("--model", help="Process only this model ID")
    parser.add_argument("--upload", action="store_true", help="Upload all files to HuggingFace")
    parser.add_argument("--force", action="store_true", help="Re-run all steps")
    args = parser.parse_args()

    # Check optimum is installed
    try:
        subprocess.run(["optimum-cli", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: optimum-cli not installed. Run: pip install optimum[exporters]")
        sys.exit(1)

    models = load_models()

    if args.model:
        models = [m for m in models if m["id"] == args.model]
        if not models:
            print(f"ERROR: model '{args.model}' not found in models.json")
            sys.exit(1)

    for model in models:
        process_model(model, upload=args.upload, force=args.force)

    print("\nDone!")


if __name__ == "__main__":
    main()
