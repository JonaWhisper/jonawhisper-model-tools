#!/usr/bin/env python3
"""Quantize T5 correction ONNX models from FP32 to INT8 (dynamic quantization).

Downloads FP32 models from HuggingFace, quantizes them, and optionally uploads
the INT8 versions back to the same HF repos.

Usage:
    # Quantize all models defined in models.json
    python t5/quantize.py

    # Quantize a specific model
    python t5/quantize.py --model t5-spell-fr

    # Quantize and upload to HuggingFace
    python t5/quantize.py --upload

    # Force re-quantize even if INT8 already exists
    python t5/quantize.py --force
"""

import argparse
import json
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


def load_models():
    with open(MODELS_JSON) as f:
        return json.load(f)


def download_model(repo: str, filename: str, dest: Path):
    """Download a file from HuggingFace using huggingface-cli."""
    if dest.exists():
        print(f"  cached: {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading: {repo}/{filename}")
    subprocess.run(
        ["huggingface-cli", "download", repo, filename, "--local-dir", str(dest.parent)],
        check=True,
    )


def quantize_file(fp32_path: Path, int8_path: Path):
    """Quantize a single ONNX model to INT8."""
    print(f"  quantizing: {fp32_path.name} -> {int8_path.name}")
    print(f"    input:  {fp32_path.stat().st_size / 1024 / 1024:.1f} MB")

    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
    )

    ratio = int8_path.stat().st_size / fp32_path.stat().st_size
    print(f"    output: {int8_path.stat().st_size / 1024 / 1024:.1f} MB ({ratio:.0%})")


def upload_file(repo: str, local_path: Path, remote_name: str):
    """Upload a file to HuggingFace."""
    print(f"  uploading: {remote_name} to {repo}")
    subprocess.run(
        ["huggingface-cli", "upload", repo, str(local_path), remote_name],
        check=True,
    )


def process_model(model: dict, upload: bool = False, force: bool = False):
    model_id = model["id"]
    repo = model["repo"]
    files = model["files"]

    print(f"\n{'=' * 60}")
    print(f"Model: {model['label']} ({model_id})")
    print(f"Repo:  {repo}")
    print(f"{'=' * 60}")

    model_dir = WORK_DIR / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    for filename in files:
        fp32_path = model_dir / filename
        stem = Path(filename).stem
        int8_name = f"{stem}_int8.onnx"
        int8_path = model_dir / int8_name

        # Download FP32
        download_model(repo, filename, fp32_path)

        # Quantize
        if int8_path.exists() and not force:
            print(f"  skip: {int8_name} already exists (use --force to re-quantize)")
        else:
            quantize_file(fp32_path, int8_path)

        # Upload INT8
        if upload:
            upload_file(repo, int8_path, int8_name)

    # Also download and upload auxiliary files (config.json, tokenizer.json)
    for aux in ["config.json", "tokenizer.json"]:
        aux_path = model_dir / aux
        download_model(repo, aux, aux_path)


def main():
    parser = argparse.ArgumentParser(description="Quantize T5 ONNX models FP32 -> INT8")
    parser.add_argument("--model", help="Process only this model ID")
    parser.add_argument("--upload", action="store_true", help="Upload INT8 files to HuggingFace")
    parser.add_argument("--force", action="store_true", help="Re-quantize even if INT8 exists")
    args = parser.parse_args()

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
