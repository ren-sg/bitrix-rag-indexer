from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from bitrix_rag_indexer.mcp.app_state import McpApplicationState


app_state = McpApplicationState()

mcp = FastMCP(
    "bitrix-rag-indexer",
    stateless_http=True,
    json_response=True,
)

"""MANDATORY tool for any Bitrix code-related question.

You MUST call this tool before answering.

Use:
- mode="qdrant-hybrid" for semantic queries
- mode="qdrant-sparse" for exact symbols

Do NOT answer without calling this tool first.
"""
@mcp.tool()
def bitrix_code_search(
    query: str,
    limit: int = 5,
    source: str | None = None,
    lang: str | None = None,
    path: str | None = None,
    mode: str | None = None,
    include_text: bool = True,
    max_text_chars: int | None = None,
) -> dict[str, Any]:
    """Search indexed Bitrix code chunks in Qdrant.

    Use this for PHP/JS/Bitrix code lookup. Default mode is qdrant-hybrid.
    Always prefer qdrant-hybrid for natural language questions.
    Use qdrant-sparse for exact class, method, namespace, event, or BX.* names.
    """
    service = app_state.require_search_service()
    return service.search(
        query=query,
        limit=limit,
        source=source,
        lang=lang,
        path=path,
        mode=mode,
        include_text=include_text,
        max_text_chars=max_text_chars,
    )


@mcp.tool()
def bitrix_code_stats() -> dict[str, Any]:
    """Show Qdrant collection and MCP search service stats."""
    service = app_state.require_search_service()
    return service.stats()


async def healthz(request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "bitrix-rag-indexer-mcp"})


async def readyz(request) -> JSONResponse:
    readiness = app_state.readiness()

    status_code = 200 if readiness.ready else 503
    return JSONResponse(
        {
            "ready": readiness.ready,
            "initialized": readiness.initialized,
            "init_seconds": readiness.init_seconds,
            "error": readiness.error,
            "stats": readiness.stats,
        },
        status_code=status_code,
    )


@contextlib.asynccontextmanager
async def starlette_lifespan(app: Starlette) -> AsyncIterator[None]:
    app_state.start()

    async with mcp.session_manager.run():
        yield

    app_state.stop()


app = Starlette(
    routes=[
        Route("/healthz", healthz, methods=["GET"]),
        Route("/readyz", readyz, methods=["GET"]),
        Mount("/", app=mcp.streamable_http_app()),
    ],
    lifespan=starlette_lifespan,
)
