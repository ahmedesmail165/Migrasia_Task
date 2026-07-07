"""Gemini embedding service with retry logic."""

from __future__ import annotations

import time
from functools import lru_cache

from google import genai
from google.genai import types

from app.config import Settings, get_settings_or_raise
from app.utils.logging import logger

_RETRYABLE_KEYWORDS = (
    "rate limit",
    "quota",
    "resource exhausted",
    "429",
    "503",
    "timeout",
    "temporarily unavailable",
    "deadline exceeded",
)

_EMBEDDING_FALLBACK_MODELS = (
    "gemini-embedding-2",
    "gemini-embedding-001",
    "text-embedding-004",
)


class EmbeddingService:
    """Generate text embeddings using the Google GenAI SDK."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings_or_raise()
        self.client = genai.Client(api_key=self.settings.gemini_api_key)
        self.model = self.settings.gemini_embedding_model
        self._active_model = self.model
        self.dimensionality = self.settings.embedding_dimensionality
        self.batch_size = self.settings.embedding_batch_size
        self.max_retries = self.settings.embedding_max_retries
        self.request_delay = self.settings.embedding_request_delay
        self._model_candidates = self._build_model_candidates()

    def _models_to_try(self) -> list[str]:
        """Return model order, preferring the last successful model."""
        ordered = [self._active_model]
        for candidate in self._model_candidates:
            if candidate not in ordered:
                ordered.append(candidate)
        return ordered

    def _build_model_candidates(self) -> list[str]:
        """Build ordered embedding model candidates with safe fallbacks."""
        candidates = [self.model, *_EMBEDDING_FALLBACK_MODELS]
        unique: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in unique:
                unique.append(candidate)
        return unique

    def _embed_config(self, model_name: str) -> types.EmbedContentConfig | None:
        """Build embedding configuration when supported by the model."""
        if model_name in {"gemini-embedding-2", "gemini-embedding-001"}:
            return types.EmbedContentConfig(
                output_dimensionality=self.dimensionality,
            )
        return None

    def _is_model_error(self, error: Exception) -> bool:
        """Detect model availability or naming errors."""
        message = str(error).lower()
        if "api_key_invalid" in message or "api key not valid" in message:
            return False
        return any(
            token in message
            for token in (
                "not found",
                "not supported",
                "invalid model",
                "unknown model",
                "does not exist",
                "404",
            )
        )

    def _is_auth_error(self, error: Exception) -> bool:
        """Detect invalid or missing API credentials."""
        message = str(error).lower()
        return "api_key_invalid" in message or "api key not valid" in message

    def _extract_embedding(self, response: types.EmbedContentResponse) -> list[float]:
        """Extract the first embedding vector from an API response."""
        if not response.embeddings:
            raise RuntimeError("Embedding API returned no vectors.")
        embedding = response.embeddings[0]
        values = getattr(embedding, "values", None)
        if values is None and isinstance(embedding, dict):
            values = embedding.get("values")
        if not values:
            raise RuntimeError("Embedding vector values are missing.")
        return [float(value) for value in values]

    def _is_retryable(self, error: Exception) -> bool:
        """Determine whether an API error should be retried."""
        message = str(error).lower()
        return any(keyword in message for keyword in _RETRYABLE_KEYWORDS)

    def _call_with_retry(self, text: str) -> list[float]:
        """Call embed_content with model fallback and exponential backoff."""
        delay = 1.0
        last_error: Exception | None = None

        for model_name in self._models_to_try():
            for attempt in range(1, self.max_retries + 1):
                try:
                    config = self._embed_config(model_name)
                    kwargs: dict = {
                        "model": model_name,
                        "contents": text,
                    }
                    if config is not None:
                        kwargs["config"] = config
                    response = self.client.models.embed_content(**kwargs)
                    if self._active_model != model_name:
                        logger.warning(
                            "Primary embedding model '%s' unavailable; using '%s'.",
                            self.model,
                            model_name,
                        )
                    self._active_model = model_name
                    return self._extract_embedding(response)
                except Exception as exc:
                    last_error = exc
                    if self._is_auth_error(exc):
                        raise RuntimeError(
                            "GEMINI_API_KEY is invalid. Create a new key at "
                            "https://aistudio.google.com/apikey and update your .env file."
                        ) from exc
                    if self._is_model_error(exc):
                        logger.warning(
                            "Embedding model '%s' unavailable: %s",
                            model_name,
                            exc,
                        )
                        break
                    if self._is_retryable(exc):
                        if attempt >= 2 and "429" in str(exc):
                            logger.warning(
                                "Rate limit on '%s'; trying next embedding model.",
                                model_name,
                            )
                            break
                        if attempt == self.max_retries:
                            break
                        logger.warning(
                            "Embedding request failed with '%s' (attempt %d/%d). "
                            "Retrying in %.1fs.",
                            model_name,
                            attempt,
                            self.max_retries,
                            delay,
                        )
                        time.sleep(delay)
                        delay = min(delay * 2, 30.0)
                        continue
                    break

        raise RuntimeError(
            "Failed to generate embedding with models "
            f"{self._models_to_try()}. Last error: {last_error}"
        ) from last_error

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Cannot embed empty text.")
        return self._call_with_retry(cleaned)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in batches."""
        if not texts:
            return []

        embeddings: list[list[float]] = []
        total = len(texts)
        for index, text in enumerate(texts, start=1):
            embeddings.append(self.embed_text(text))
            if index < total and self.request_delay > 0:
                time.sleep(self.request_delay)
            if index % 25 == 0:
                logger.info(
                    "Embedded %d/%d chunks using model '%s'.",
                    index,
                    total,
                    self._active_model,
                )
        return embeddings


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Return a cached embedding service instance."""
    return EmbeddingService()
