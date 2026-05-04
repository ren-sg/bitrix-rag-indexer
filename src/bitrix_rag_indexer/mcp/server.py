from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from bitrix_rag_indexer.mcp.search_service import BitrixCodeSearchService
from bitrix_rag_indexer.mcp.settings import McpServerSettings


@dataclass
class McpAppContext:
    search_service: BitrixCodeSearchService


@contextlib.asynccontextmanager
async def mcp_lifespan(server: FastMCP) -> AsyncIterator[McpAppContext]:
    settings = McpServerSettings.from_env()
    search_service = BitrixCodeSearchService(settings)
    yield McpAppContext(search_service=search_service)


mcp = FastMCP(
    "bitrix-rag-indexer",
    stateless_http=True,
    json_response=True,
    lifespan=mcp_lifespan,
)


@mcp.tool()
def bitrix_code_search(
    query: str,
    ctx: Context,
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
    Filters:
    - source: project_local, manual, bitrix_im, etc.
    - lang: php, js, javascript, ts, markdown, etc.
    - path: text filter over rel_path.
    """
    service = ctx.request_context.lifespan_context.search_service
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
def bitrix_code_stats(ctx: Context) -> dict[str, Any]:
    """Show Qdrant collection and MCP search service stats."""
    service = ctx.request_context.lifespan_context.search_service
    return service.stats()


async def healthz(request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "bitrix-rag-indexer-mcp"})


@contextlib.asynccontextmanager
async def starlette_lifespan(app: Starlette) -> AsyncIterator[None]:
    async with mcp.session_manager.run():
        yield


app = Starlette(
    routes=[
        Route("/healthz", healthz, methods=["GET"]),
        Mount("/", app=mcp.streamable_http_app()),
    ],
    lifespan=starlette_lifespan,
)
