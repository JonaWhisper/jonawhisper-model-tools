#!/usr/bin/env bash
# Build a pruned trigram KenLM model from a Wikipedia dump.
#
# Usage:
#   ./train.sh fr          # French
#   ./train.sh en          # English
#   ./train.sh fr 4        # French 4-gram (default: 3)
#
# Prerequisites:
#   brew install kenlm     # or build from source (github.com/kpu/kenlm)
#   pip install wikiextractor
#
# Output: {lang}.binary (KenLM trie format, 8-bit quantized)

set -euo pipefail

LANG="${1:?Usage: $0 <lang> [order]}"
ORDER="${2:-3}"

WORKDIR="work/${LANG}"
mkdir -p "$WORKDIR"

WIKI_DUMP="https://dumps.wikimedia.org/${LANG}wiki/latest/${LANG}wiki-latest-pages-articles.xml.bz2"
DUMP_FILE="${WORKDIR}/${LANG}wiki.xml.bz2"
TEXT_DIR="${WORKDIR}/text"
CORPUS="${WORKDIR}/corpus.txt"
ARPA="${WORKDIR}/${LANG}.arpa"
BINARY="${LANG}.binary"

# -- Step 1: Download Wikipedia dump --
if [ ! -f "$DUMP_FILE" ]; then
    echo "==> Downloading ${LANG} Wikipedia dump..."
    wget -c "$WIKI_DUMP" -O "$DUMP_FILE"
else
    echo "==> Dump already downloaded: $DUMP_FILE"
fi

# -- Step 2: Extract text --
if [ ! -d "$TEXT_DIR" ]; then
    echo "==> Extracting text with WikiExtractor..."
    python3 -m wikiextractor.WikiExtractor "$DUMP_FILE" \
        --no-templates \
        --processes 4 \
        -o "$TEXT_DIR"
else
    echo "==> Text already extracted: $TEXT_DIR"
fi

# -- Step 3: Clean and normalize --
if [ ! -f "$CORPUS" ]; then
    echo "==> Cleaning and normalizing corpus..."
    find "$TEXT_DIR" -name "wiki_*" -exec cat {} + | \
        python3 "$(dirname "$0")/normalize.py" > "$CORPUS"

    LINES=$(wc -l < "$CORPUS")
    echo "    Corpus: ${LINES} sentences"
else
    echo "==> Corpus already prepared: $CORPUS"
fi

# -- Step 4: Train n-gram model --
if [ ! -f "$ARPA" ]; then
    echo "==> Training ${ORDER}-gram model with KenLM..."
    # --prune 0 0 1 : keep all unigrams and bigrams, prune singleton trigrams
    # This dramatically reduces size while keeping quality
    lmplz -o "$ORDER" --prune 0 0 1 < "$CORPUS" > "$ARPA"

    ARPA_SIZE=$(du -h "$ARPA" | cut -f1)
    echo "    ARPA model: ${ARPA_SIZE}"
else
    echo "==> ARPA model already trained: $ARPA"
fi

# -- Step 5: Convert to binary trie with quantization --
echo "==> Converting to binary trie (8-bit quantization)..."
# -q 8 : 8-bit quantization for probabilities (~60% smaller)
# -b 8 : 8-bit quantization for backoff weights
# trie : trie data structure (smaller than probing hash)
build_binary trie -q 8 -b 8 "$ARPA" "$BINARY"

BINARY_SIZE=$(du -h "$BINARY" | cut -f1)
BINARY_BYTES=$(stat -f%z "$BINARY" 2>/dev/null || stat -c%s "$BINARY" 2>/dev/null)
echo ""
echo "==> Done!"
echo "    Output: ${BINARY} (${BINARY_SIZE})"
echo "    Size (bytes): ${BINARY_BYTES}"
echo ""
echo "    Upload with:"
echo "    huggingface-cli upload JonaWhisper/kenlm-models ${BINARY} ${BINARY}"
