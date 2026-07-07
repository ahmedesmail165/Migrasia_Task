"""Unit tests for retrieval depth resolution."""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings, resolve_retrieval_depth


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(monkeypatch) -> Settings:
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    monkeypatch.setenv("DEFAULT_TOP_K", "8")
    monkeypatch.setenv("MIN_RETRIEVAL_K", "8")
    monkeypatch.setenv("MAX_RETRIEVAL_K", "20")
    monkeypatch.setenv("MAX_DISPLAY_SOURCES", "10")
    get_settings.cache_clear()
    return get_settings()


def test_default_depth_when_top_k_omitted(settings: Settings) -> None:
    retrieval_k, display_k = resolve_retrieval_depth(None, settings)
    assert retrieval_k == 8
    assert display_k == 8


def test_low_top_k_preserves_internal_depth(settings: Settings) -> None:
    retrieval_k, display_k = resolve_retrieval_depth(1, settings)
    assert retrieval_k == 8
    assert display_k == 1


def test_high_top_k_caps_display(settings: Settings) -> None:
    retrieval_k, display_k = resolve_retrieval_depth(15, settings)
    assert retrieval_k == 15
    assert display_k == 10
