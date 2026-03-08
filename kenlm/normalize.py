"""Normalize WikiExtractor output for KenLM training.

Reads from stdin, writes one sentence per line to stdout.
- Strips XML/HTML tags
- Splits on sentence boundaries (. ? ! followed by space+uppercase or newline)
- Lowercases everything (KenLM works best with lowercased input)
- Removes lines that are too short (<3 words) or too long (>200 words)
- Strips URLs, email addresses, and non-text artifacts
"""

import re
import sys


# Patterns to strip
TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
PARENS_RE = re.compile(r"\([^)]*\)")  # Remove parenthetical content
MULTI_SPACE = re.compile(r"\s+")
# Sentence boundary: . or ? or ! followed by space and uppercase, or end of line
SENT_SPLIT = re.compile(r"(?<=[.?!])\s+(?=[A-ZÀ-ÖØ-Þ])")


def normalize_line(line: str) -> list[str]:
    """Process one input line, return list of normalized sentences."""
    # Strip tags and artifacts
    line = TAG_RE.sub("", line)
    line = URL_RE.sub("", line)
    line = EMAIL_RE.sub("", line)

    # Skip empty/metadata lines
    stripped = line.strip()
    if not stripped or stripped.startswith("CATEGORIES:"):
        return []

    # Split into sentences
    sentences = SENT_SPLIT.split(stripped)

    results = []
    for sent in sentences:
        # Lowercase
        sent = sent.lower().strip()
        # Remove parenthetical content (often citations/references)
        sent = PARENS_RE.sub("", sent)
        # Normalize whitespace
        sent = MULTI_SPACE.sub(" ", sent).strip()
        # Remove trailing punctuation for cleaner n-grams
        sent = sent.rstrip(".?!,;:")

        # Filter: 3-200 words, must contain at least one letter
        words = sent.split()
        if 3 <= len(words) <= 200 and any(c.isalpha() for c in sent):
            results.append(sent)

    return results


def main():
    for line in sys.stdin:
        for sent in normalize_line(line):
            print(sent)


if __name__ == "__main__":
    main()
