"""Pydantic request and response schemas."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat endpoint request body."""

    question: str = Field(..., min_length=1, description="User question")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description=(
            "Number of source citations to return. The system always retrieves "
            "at least MIN_RETRIEVAL_K chunks internally for answer quality."
        ),
    )


class SourceCitation(BaseModel):
    """A retrieved source used to ground the answer."""

    source_file: str
    page_start: int
    page_end: int
    score: float


class ChatResponse(BaseModel):
    """Chat endpoint response body."""

    answer: str
    sources: list[SourceCitation]
    confidence: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    vector_store_loaded: bool
    chunks_count: int


class IngestResponse(BaseModel):
    """Document ingestion response."""

    files_processed: int
    pages_processed: int
    chunks_created: int
    index_path: str
    chunks_path: str


class DocumentSourceInfo(BaseModel):
    """Summary of an indexed document."""

    source_file: str
    chunk_count: int


class SourcesResponse(BaseModel):
    """List of indexed documents."""

    documents: list[DocumentSourceInfo]
    total_chunks: int
