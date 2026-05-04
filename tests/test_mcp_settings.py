from __future__ import annotations

from pathlib import Path

from bitrix_rag_indexer.mcp.settings import McpServerSettings


def test_mcp_settings_from_env(monkeypatch) -> None:
    monkeypatch.setenv("BITRIX_RAG_PROFILE", "mvp")
    monkeypatch.setenv("BITRIX_RAG_CONFIG_DIR", "/app/configs")
    monkeypatch.setenv("BITRIX_RAG_QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("BITRIX_RAG_SEARCH_MODE", "qdrant-hybrid")
    monkeypatch.setenv("BITRIX_RAG_DEFAULT_LIMIT", "7")
    monkeypatch.setenv("BITRIX_RAG_MAX_LIMIT", "15")
    monkeypatch.setenv("BITRIX_RAG_MAX_TEXT_CHARS", "1234")

    settings = McpServerSettings.from_env()

    assert settings.profile == "mvp"
    assert settings.config_dir == Path("/app/configs")
    assert settings.qdrant_url == "http://qdrant:6333"
    assert settings.default_mode == "qdrant-hybrid"
    assert settings.default_limit == 7
    assert settings.max_limit == 15
    assert settings.max_text_chars == 1234
