# KenLM Language Models for JonaWhisper

Pruned trigram language models trained on Wikipedia, used for context-aware spell correction.

## Prerequisites

```bash
# KenLM tools (lmplz, build_binary)
brew install kenlm
# or build from source: https://github.com/kpu/kenlm

# Python dependencies
pip install -r kenlm/requirements.txt
```

## Usage

```bash
# Build a model
cd kenlm
./train.sh fr    # French trigram (~50-100 MB)
./train.sh en    # English trigram (~60-120 MB)

# Upload to HuggingFace
python upload.py fr.binary fr
python upload.py en.binary en
```

`train.sh` will:
1. Download the Wikipedia dump for the language
2. Extract and normalize text (one sentence per line, lowercased)
3. Train a pruned trigram model with KenLM (`--prune 0 0 1`)
4. Convert to binary trie format with 8-bit quantization

Output: `{lang}.binary`

## Adding a language

1. Add the language code to `languages.json` (e.g. `["fr", "en", "de"]`)
2. Trigger the `Build KenLM models` workflow

That's it. The pipeline works for any language that has a Wikipedia dump.

## CI

The `Build KenLM models` workflow reads `languages.json`, builds KenLM from source, trains one model per language in parallel, and uploads to HuggingFace. Triggered manually from the Actions tab.

## Model details

- **Order**: 3 (trigram)
- **Pruning**: `--prune 0 0 1` (remove singleton trigrams)
- **Quantization**: 8-bit probabilities + 8-bit backoff weights
- **Format**: KenLM trie binary (mmap-compatible)
- **Corpus**: Full Wikipedia dump per language
- **Lowercased**: Yes (spell correction operates on lowercase)

## How it works in JonaWhisper

The `jona-engine-lm` crate loads these models via FFI to the vendored KenLM C++ code.
When the user has both a SymSpell dictionary and a KenLM model for a language:

1. SymSpell generates correction candidates for unknown words
2. KenLM scores each candidate in trigram context
3. The candidate with the highest language model probability wins

Without a KenLM model, SymSpell falls back to frequency-only correction.
