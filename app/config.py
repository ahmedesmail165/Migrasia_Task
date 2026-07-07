"""Application configuration using pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    gemini_chat_model: str = Field(
        default="gemini-2.5-flash",
        alias="GEMINI_CHAT_MODEL",
    )
    gemini_embedding_model: str = Field(
        default="gemini-embedding-2",
        alias="GEMINI_EMBEDDING_MODEL",
    )
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    processed_dir: Path = Field(default=Path("processed"), alias="PROCESSED_DIR")
    vector_store_dir: Path = Field(
        default=Path("vector_store"),
        alias="VECTOR_STORE_DIR",
    )
    chunk_size: int = Field(default=1200, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    default_top_k: int = Field(
        default=8,
        validation_alias=AliasChoices("DEFAULT_TOP_K", "TOP_K"),
    )
    min_retrieval_k: int = Field(default=8, alias="MIN_RETRIEVAL_K")
    max_retrieval_k: int = Field(default=20, alias="MAX_RETRIEVAL_K")
    max_display_sources: int = Field(default=10, alias="MAX_DISPLAY_SOURCES")
    min_confidence_score: float = Field(
        default=0.35,
        alias="MIN_CONFIDENCE_SCORE",
    )
    embedding_dimensionality: int = Field(
        default=768,
        alias="EMBEDDING_DIMENSIONALITY",
    )
    embedding_batch_size: int = Field(default=16, alias="EMBEDDING_BATCH_SIZE")
    embedding_max_retries: int = Field(default=5, alias="EMBEDDING_MAX_RETRIES")
    embedding_request_delay: float = Field(
        default=1.5,
        alias="EMBEDDING_REQUEST_DELAY",
    )

    @field_validator("gemini_api_key")
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        """Ensure GEMINI_API_KEY is present and not a placeholder."""
        cleaned = value.strip()
        if not cleaned or cleaned == "your_gemini_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY is missing or invalid. "
                "Set a valid API key in your .env file."
            )
        return cleaned

    @property
    def top_k(self) -> int:
        """Backward-compatible alias for default_top_k."""
        return self.default_top_k

    @property
    def chunks_path(self) -> Path:
        """Path to the processed chunks JSONL file."""
        return self.processed_dir / "chunks.jsonl"

    @property
    def faiss_index_path(self) -> Path:
        """Path to the FAISS index file."""
        return self.vector_store_dir / "index.faiss"

    @property
    def metadata_path(self) -> Path:
        """Path to the vector store metadata JSON file."""
        return self.vector_store_dir / "metadata.json"


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()  # type: ignore[call-arg]


def resolve_retrieval_depth(
    requested_top_k: int | None,
    settings: Settings,
) -> tuple[int, int]:
    """
    Map a user-facing top_k to internal retrieval depth and display count.

    ``retrieval_k`` is used for hybrid search, answerability checks, and LLM
    context. ``display_k`` limits how many sources are returned to the client.
    """
    requested = requested_top_k if requested_top_k is not None else settings.default_top_k
    retrieval_k = min(
        max(requested, settings.min_retrieval_k),
        settings.max_retrieval_k,
    )
    display_k = min(requested, settings.max_display_sources)
    return retrieval_k, display_k


def get_settings_or_raise() -> Settings:
    """Load settings and raise a clear error if configuration is invalid."""
    try:
        return get_settings()
    except Exception as exc:
        raise RuntimeError(
            "Configuration error: ensure GEMINI_API_KEY is set in your .env file. "
            f"Details: {exc}"
        ) from exc
