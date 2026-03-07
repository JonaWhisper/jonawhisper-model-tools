#!/usr/bin/env python3
"""Convert T5 correction models from HuggingFace (PyTorch) to ONNX.

Downloads the original PyTorch models from their source repos, exports them
to ONNX via optimum-cli, and optionally uploads the ONNX files to the
JonaWhisper HuggingFace repos.

Requires: pip install optimum[exporters] onnx huggingface-hub

Usage:
    # Convert all models defined in models.json
    python t5/convert.py

    # Convert a specific model
    python t5/convert.py --model t5-spell-fr

    # Convert and upload to HuggingFace
    python t5/convert.py --upload

    # Force re-convert even if ONNX already exists
    python t5/convert.py --force
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MODELS_JSON = SCRIPT_DIR / "models.json"
WORK_DIR = SCRIPT_DIR / "work"

# Files to upload after conversion
UPLOAD_FILES = [
    "encoder_model.onnx",
    "decoder_model.onnx",
    "config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
]


def load_models():
    with open(MODELS_JSON) as f:
        return json.load(f)


def convert_model(source: str, output_dir: Path):
    """Convert a PyTorch model to ONNX using optimum-cli."""
    print(f"  converting: {source} → ONNX")
    subprocess.run(
        [
            sys.executable, "-m", "optimum.exporters.onnx",
            "--model", source,
            "--task", "text2text-generation",
            str(output_dir),
        ],
        check=True,
    )

    # Show output sizes
    for f in sorted(output_dir.glob("*.onnx")):
        print(f"    {f.name}: {f.stat().st_size / 1024 / 1024:.1f} MB")


def upload_file(repo: str, local_path: Path, remote_name: str):
    """Upload a file to HuggingFace."""
    print(f"  uploading: {remote_name} to {repo}")
    subprocess.run(
        ["huggingface-cli", "upload", repo, str(local_path), remote_name],
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

    onnx_dir = WORK_DIR / model_id / "onnx"

    # Check if already converted
    encoder = onnx_dir / "encoder_model.onnx"
    decoder = onnx_dir / "decoder_model.onnx"

    if encoder.exists() and decoder.exists() and not force:
        print(f"  skip: ONNX already exists (use --force to re-convert)")
        for f in sorted(onnx_dir.glob("*.onnx")):
            print(f"    {f.name}: {f.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        if onnx_dir.exists():
            shutil.rmtree(onnx_dir)
        onnx_dir.mkdir(parents=True, exist_ok=True)
        convert_model(source, onnx_dir)

    # Upload
    if upload:
        for name in UPLOAD_FILES:
            f = onnx_dir / name
            if f.exists():
                upload_file(repo, f, name)
            else:
                print(f"  skip upload: {name} not found")


def main():
    parser = argparse.ArgumentParser(description="Convert T5 models from PyTorch to ONNX")
    parser.add_argument("--model", help="Process only this model ID")
    parser.add_argument("--upload", action="store_true", help="Upload ONNX files to HuggingFace")
    parser.add_argument("--force", action="store_true", help="Re-convert even if ONNX exists")
    args = parser.parse_args()

    # Check optimum is installed
    try:
        import optimum.exporters.onnx  # noqa: F401
    except ImportError:
        print("ERROR: optimum not installed. Run: pip install optimum[exporters]")
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
