#!/usr/bin/env python3
"""Upload a KenLM binary model to HuggingFace.

Creates the repo if it doesn't exist, then uploads the binary file.

Usage:
    python upload.py fr.binary fr
    python upload.py en.binary en
"""

import argparse
import sys
from pathlib import Path

REPO_ID = "JonaWhisper/kenlm-models"

MODEL_CARD = """\
---
tags:
  - kenlm
  - n-gram
  - language-model
  - spell-correction
license: lgpl-2.1
---

# KenLM Language Models for JonaWhisper

Pruned trigram language models trained on Wikipedia, used for context-aware spell correction
in [JonaWhisper](https://github.com/jplot/jona-whisper).

## Models

| File | Language | Order | Pruning | Quantization |
|------|----------|-------|---------|--------------|
| `fr.binary` | French | 3-gram | `--prune 0 0 1` | 8-bit |
| `en.binary` | English | 3-gram | `--prune 0 0 1` | 8-bit |

## Training

Trained from full Wikipedia dumps using KenLM's `lmplz` + `build_binary trie`.
See [jonawhisper-model-tools/kenlm](https://github.com/JonaWhisper/jonawhisper-model-tools/tree/main/kenlm).

## License

KenLM is LGPL-2.1. Wikipedia text is CC BY-SA 3.0.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", help="Path to .binary file")
    parser.add_argument("lang", help="Language code (fr, en)")
    args = parser.parse_args()

    binary = Path(args.binary)
    if not binary.exists():
        print(f"ERROR: {binary} not found")
        sys.exit(1)

    size = binary.stat().st_size
    print(f"Uploading {binary.name} ({size / 1024 / 1024:.1f} MB) to {REPO_ID}")

    from huggingface_hub import HfApi
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.repo_info(repo_id=REPO_ID, repo_type="model")
        print(f"  Repo {REPO_ID} exists")
    except Exception:
        print(f"  Creating repo {REPO_ID}")
        api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    # Upload model card (once)
    try:
        api.upload_file(
            path_or_fileobj=MODEL_CARD.encode(),
            path_in_repo="README.md",
            repo_id=REPO_ID,
        )
        print("  Uploaded README.md")
    except Exception as e:
        print(f"  README.md upload skipped: {e}")

    # Upload binary
    api.upload_file(
        path_or_fileobj=str(binary),
        path_in_repo=binary.name,
        repo_id=REPO_ID,
    )
    print(f"  Uploaded {binary.name}")
    print(f"  Size: {size} bytes")
    print("Done!")


if __name__ == "__main__":
    main()
