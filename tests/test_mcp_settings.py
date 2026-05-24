from __future__ import annotations

from pathlib import Path

from bitrix_rag_indexer.mcp.settings import McpServerSettings
from bitrix_rag_indexer.search.result_middleware import parse_use_abs_path


def test_mcp_settings_from_env(monkeypatch) -> None:
    monkeypatch.setenv("BITRIX_RAG_CONFIG_DIR", "/app/configs")
    monkeypatch.setenv("BITRIX_RAG_QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("USE_ABS_PATH", "true")
    monkeypatch.setenv("BITRIX_RAG_SEARCH_MODE", "qdrant-hybrid")
    monkeypatch.setenv("BITRIX_RAG_DEFAULT_LIMIT", "7")
    monkeypatch.setenv("BITRIX_RAG_MAX_LIMIT", "15")
    monkeypatch.setenv("BITRIX_RAG_MAX_TEXT_CHARS", "1234")

    settings = McpServerSettings.from_env()

    assert settings.config_dir == Path("/app/configs")
    assert settings.qdrant_url == "http://qdrant:6333"
    assert settings.use_abs_path is True
    assert settings.default_mode == "qdrant-hybrid"
    assert settings.default_limit == 7
    assert settings.max_limit == 15
    assert settings.max_text_chars == 1234


def test_mcp_settings_use_abs_path_defaults_false(monkeypatch) -> None:
    monkeypatch.delenv("USE_ABS_PATH", raising=False)

    settings = McpServerSettings.from_env()

    assert settings.use_abs_path is False


def test_parse_use_abs_path_accepts_common_truthy_values() -> None:
    assert parse_use_abs_path("true") is True
    assert parse_use_abs_path("1") is True
    assert parse_use_abs_path("yes") is True
    assert parse_use_abs_path("false") is False
    assert parse_use_abs_path("") is False
