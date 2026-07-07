"""Document loading from the data directory."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

import fitz

from app.utils.logging import logger

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm"}


@dataclass
class PageDocument:
    """A single page or section extracted from a source file."""

    source_file: str
    page_number: int
    document_type: str
    source_path: str
    text: str


def _strip_html_tags(raw_html: str) -> str:
    """Remove HTML tags while preserving readable text."""
    without_scripts = re.sub(
        r"<(script|style)[^>]*>.*?</\1>",
        " ",
        raw_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    without_tags = re.sub(r"<[^>]+>", " ", without_scripts)
    return html.unescape(without_tags)


def _load_pdf(file_path: Path) -> list[PageDocument]:
    """Extract text from a PDF page by page."""
    pages: list[PageDocument] = []
    doc = fitz.open(file_path)
    try:
        for index in range(len(doc)):
            page = doc.load_page(index)
            text = page.get_text("text").strip()
            if not text:
                logger.debug("Skipping empty page %s in %s", index + 1, file_path.name)
                continue
            pages.append(
                PageDocument(
                    source_file=file_path.name,
                    page_number=index + 1,
                    document_type="pdf",
                    source_path=str(file_path),
                    text=text,
                )
            )
    finally:
        doc.close()
    return pages


def _load_text_file(file_path: Path, document_type: str) -> list[PageDocument]:
    """Load a plain text, markdown, or HTML file as a single page."""
    raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    if document_type == "html":
        raw_text = _strip_html_tags(raw_text)
    text = raw_text.strip()
    if not text:
        return []
    return [
        PageDocument(
            source_file=file_path.name,
            page_number=1,
            document_type=document_type,
            source_path=str(file_path),
            text=text,
        )
    ]


def load_documents(data_dir: Path) -> list[PageDocument]:
    """
    Scan the data directory and load supported documents.

    Returns a list of page-level documents with metadata.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            "Create it and add PDF or text documents."
        )

    documents: list[PageDocument] = []
    files = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        logger.warning("No supported documents found in %s", data_dir)

    for file_path in files:
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".pdf":
                pages = _load_pdf(file_path)
            elif suffix in {".txt"}:
                pages = _load_text_file(file_path, "txt")
            elif suffix in {".md"}:
                pages = _load_text_file(file_path, "md")
            elif suffix in {".html", ".htm"}:
                pages = _load_text_file(file_path, "html")
            else:
                continue

            documents.extend(pages)
            logger.info(
                "Loaded %s (%d page(s))",
                file_path.name,
                len(pages),
            )
        except Exception as exc:
            logger.error("Failed to load %s: %s", file_path, exc)

    return documents
