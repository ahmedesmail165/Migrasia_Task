"""Document chunking for retrieval."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.services.document_loader import PageDocument
from app.services.text_cleaner import clean_text
from app.utils.logging import logger

SECTION_PATTERN = re.compile(
    r"(?m)^(?:"
    r"(?:\d+(?:\.\d+)*)\s+.+"
    r"|(?:Section|Chapter|Part|Article)\s+\d+.+"
    r"|(?:第\s*\d+\s*章|第\s*\d+\s*條).+"
    r")"
)


@dataclass
class Chunk:
    """A retrieval-ready text chunk with metadata."""

    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    text: str
    text_hash: str


def _hash_text(text: str) -> str:
    """Return a stable SHA-256 hash for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_chunk_id(source_file: str, page_start: int, page_end: int, text_hash: str) -> str:
    """Create a deterministic chunk identifier."""
    raw = f"{source_file}:{page_start}:{page_end}:{text_hash[:12]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split long text using paragraph boundaries when possible."""
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return [text[:chunk_size]]

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                if end < len(paragraph):
                    split_at = paragraph.rfind(" ", start, end)
                    if split_at > start + int(chunk_size * 0.5):
                        end = split_at
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(paragraph):
                    break
                start = max(end - chunk_overlap, start + 1)
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())

    return _apply_overlap(chunks, chunk_overlap)


def _apply_overlap(chunks: list[str], chunk_overlap: int) -> list[str]:
    """Add textual overlap between adjacent chunks."""
    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = []
    for index, chunk in enumerate(chunks):
        if index == 0:
            overlapped.append(chunk)
            continue
        previous = chunks[index - 1]
        prefix = previous[-chunk_overlap:] if len(previous) > chunk_overlap else previous
        merged = f"{prefix}\n{chunk}".strip()
        overlapped.append(merged)
    return overlapped


def _chunk_page_group(
    pages: list[PageDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Chunk a group of pages from the same source file."""
    if not pages:
        return []

    combined_text = "\n\n".join(page.text for page in pages)
    sections = SECTION_PATTERN.split(combined_text)
    section_chunks: list[str] = []

    if len(sections) <= 1:
        section_chunks = _split_long_text(combined_text, chunk_size, chunk_overlap)
    else:
        buffer = ""
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) < max(120, chunk_size // 4):
                buffer = f"{buffer}\n\n{section}".strip() if buffer else section
                continue
            if buffer:
                section = f"{buffer}\n\n{section}".strip()
                buffer = ""
            section_chunks.extend(_split_long_text(section, chunk_size, chunk_overlap))
        if buffer:
            section_chunks.extend(_split_long_text(buffer, chunk_size, chunk_overlap))

    page_start = min(page.page_number for page in pages)
    page_end = max(page.page_number for page in pages)
    source_file = pages[0].source_file

    chunks: list[Chunk] = []
    for text in section_chunks:
        text = text.strip()
        if not text:
            continue
        text_hash = _hash_text(text)
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(source_file, page_start, page_end, text_hash),
                source_file=source_file,
                page_start=page_start,
                page_end=page_end,
                text=text,
                text_hash=text_hash,
            )
        )
    return chunks


def chunk_documents(
    pages: list[PageDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """
    Convert page documents into deduplicated retrieval chunks.

    Groups consecutive pages from the same file before chunking to avoid
    splitting very short sections when possible.
    """
    cleaned_pages: list[PageDocument] = []
    for page in pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue
        cleaned_pages.append(
            PageDocument(
                source_file=page.source_file,
                page_number=page.page_number,
                document_type=page.document_type,
                source_path=page.source_path,
                text=cleaned,
            )
        )

    grouped: dict[str, list[PageDocument]] = {}
    for page in cleaned_pages:
        grouped.setdefault(page.source_file, []).append(page)

    all_chunks: list[Chunk] = []
    seen_hashes: set[str] = set()

    for source_file in sorted(grouped):
        file_pages = sorted(grouped[source_file], key=lambda item: item.page_number)
        batch: list[PageDocument] = []
        batch_chars = 0
        max_group_chars = chunk_size * 3

        for page in file_pages:
            page_len = len(page.text)
            if batch and batch_chars + page_len > max_group_chars:
                all_chunks.extend(_chunk_page_group(batch, chunk_size, chunk_overlap))
                batch = [page]
                batch_chars = page_len
            else:
                batch.append(page)
                batch_chars += page_len

        if batch:
            all_chunks.extend(_chunk_page_group(batch, chunk_size, chunk_overlap))

    deduplicated: list[Chunk] = []
    for chunk in all_chunks:
        if chunk.text_hash in seen_hashes:
            logger.debug("Skipping duplicate chunk from %s", chunk.source_file)
            continue
        seen_hashes.add(chunk.text_hash)
        deduplicated.append(chunk)

    logger.info("Created %d unique chunks from %d pages", len(deduplicated), len(cleaned_pages))
    return deduplicated


def save_chunks(chunks: list[Chunk], output_path: Path) -> None:
    """Persist chunks to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {
                        "chunk_id": chunk.chunk_id,
                        "source_file": chunk.source_file,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "text": chunk.text,
                        "text_hash": chunk.text_hash,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_chunks(chunks_path: Path) -> list[Chunk]:
    """Load chunks from a JSONL file."""
    if not chunks_path.exists():
        return []

    chunks: list[Chunk] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            chunks.append(Chunk(**payload))
    return chunks
