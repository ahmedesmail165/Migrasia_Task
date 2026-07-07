"""Text cleaning for legal and regulatory documents."""

from __future__ import annotations

import re
import unicodedata

# Common PDF header/footer and navigation noise patterns.
_NOISE_PATTERNS = [
    re.compile(r"^\s*page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*www\.[a-z0-9.-]+\.[a-z]{2,}\s*$", re.IGNORECASE),
    re.compile(r"^\s*labor\.gov\.hk.*$", re.IGNORECASE),
    re.compile(r"^\s*labour\.gov\.hk.*$", re.IGNORECASE),
    re.compile(r"^\s*immd\.gov\.hk.*$", re.IGNORECASE),
]


def _remove_broken_characters(text: str) -> str:
    """Replace control characters and common mojibake artifacts."""
    normalized = unicodedata.normalize("NFKC", text)
    cleaned_chars: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category.startswith("C") and char not in "\n\t":
            continue
        if char in {"\ufffd", "\u00ad"}:
            continue
        cleaned_chars.append(char)
    return "".join(cleaned_chars)


def clean_text(text: str) -> str:
    """
    Clean extracted document text while preserving legal content.

    Normalizes whitespace, removes obvious noise, and keeps section numbers,
    penalties, amounts, dates, and official names intact.
    """
    text = _remove_broken_characters(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if any(pattern.match(stripped) for pattern in _NOISE_PATTERNS):
            continue
        lines.append(stripped)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
