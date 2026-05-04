from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from bitrix_rag_indexer.mcp.search_service import BitrixCodeSearchService
from bitrix_rag_indexer.mcp.settings import McpServerSettings


@dataclass
class McpReadiness:
    ready: bool
    initialized: bool
    init_seconds: float | None
    error: str | None
    stats: dict[str, Any] | None


class McpApplicationState:
    def __init__(self) -> None:
        self._search_service: BitrixCodeSearchService | None = None
        self._init_seconds: float | None = None
        self._startup_error: str | None = None

    def start(self) -> None:
        started_at = perf_counter()

        try:
            settings = McpServerSettings.from_env()
            self._search_service = BitrixCodeSearchService(settings)
            self._init_seconds = perf_counter() - started_at
            self._startup_error = None
        except Exception as exc:
            self._search_service = None
            self._init_seconds = perf_counter() - started_at
            self._startup_error = f"{type(exc).__name__}: {exc}"
            raise

    def stop(self) -> None:
        self._search_service = None

    def require_search_service(self) -> BitrixCodeSearchService:
        if self._search_service is None:
            raise RuntimeError("MCP search service is not initialized")

        return self._search_service

    def readiness(self) -> McpReadiness:
        if self._search_service is None:
            return McpReadiness(
                ready=False,
                initialized=False,
                init_seconds=self._init_seconds,
                error=self._startup_error,
                stats=None,
            )

        return McpReadiness(
            ready=True,
            initialized=True,
            init_seconds=self._init_seconds,
            error=None,
            stats=self._search_service.stats(),
        )
