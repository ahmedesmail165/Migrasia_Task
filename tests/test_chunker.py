"""Tests for the chunker service."""

from app.services.chunker import Chunk, chunk_documents, save_chunks, load_chunks
from app.services.document_loader import PageDocument


def _make_page(source_file: str, page_number: int, text: str) -> PageDocument:
    return PageDocument(
        source_file=source_file,
        page_number=page_number,
        document_type="txt",
        source_path=f"data/{source_file}",
        text=text,
    )


def test_chunker_creates_non_empty_chunks() -> None:
    pages = [
        _make_page("guide.txt", 1, "Section 1. Worker rights.\n\n" + ("Workers have rights. " * 40)),
        _make_page("guide.txt", 2, "Section 2. Leave entitlements.\n\n" + ("Leave rules apply. " * 40)),
    ]
    chunks = chunk_documents(pages, chunk_size=200, chunk_overlap=40)
    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(chunk.chunk_id for chunk in chunks)


def test_chunk_overlap_is_applied() -> None:
    long_text = "Paragraph one about wages and payment schedules. " * 20
    long_text += "\n\nParagraph two about rest days and statutory holidays. " * 20
    pages = [_make_page("wages.txt", 1, long_text)]
    chunks = chunk_documents(pages, chunk_size=250, chunk_overlap=60)
    assert len(chunks) >= 2
    second_start = chunks[1].text[:60]
    first_end = chunks[0].text[-60:]
    assert any(token in second_start for token in first_end.split() if len(token) > 4)


def test_duplicate_chunks_are_removed(tmp_path) -> None:
    duplicate_text = "Repeated legal clause about passport retention. " * 10
    pages = [
        _make_page("a.txt", 1, duplicate_text),
        _make_page("b.txt", 1, duplicate_text),
    ]
    chunks = chunk_documents(pages, chunk_size=500, chunk_overlap=0)
    hashes = [chunk.text_hash for chunk in chunks]
    assert len(hashes) == len(set(hashes))

    output = tmp_path / "chunks.jsonl"
    save_chunks(chunks, output)
    loaded = load_chunks(output)
    assert len(loaded) == len(chunks)
