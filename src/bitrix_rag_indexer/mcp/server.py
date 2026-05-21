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

@mcp.tool()
def bitrix_semantic_search(
    query: str,
    limit: int = 3,
    project: str | None = None,
    lang: str | None = "php",
    path: str | None = None,
    php_namespace: str | None = None,
    php_class: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """Search for ACTUAL CODE implementations or semantic queries.

    BEST TOOL for fetching exact code snippets to solve problems.
    Returns the FULL code text. 
    IMPORTANT: Keep `limit` low (e.g., 2-4) to avoid flooding your context window!
    If you need to broadly explore many files or find where a method is used, use `bitrix_code_locator` instead.
    
    Use:
    - mode="qdrant-hybrid" (default) for semantic natural language queries.
    - mode="qdrant-sparse" for exact symbols (class names, exact function names).
    
    Optional Filters:
    - limit: Max number of results (default 3). Keep it low to save context!
    - lang: Filter by programming language (default "php", e.g., "javascript", "vue", "markdown").
    - path: Filter by file path substring (e.g., "local/components", "modules/sale").
    - project: Filter by project name.
    - php_namespace: Filter by exact PHP namespace (e.g., "Bitrix\\Sale").
    - php_class: Filter by exact PHP class or interface name (e.g., "Basket").
    
    TODO: Implement cross-encoder reranker in the future to improve top-k accuracy before returning.
    """
    service = app_state.require_search_service()
    return service.search(
        query=query,
        limit=limit,
        project=project,
        lang=lang,
        path=path,
        php_namespace=php_namespace,
        php_class=php_class,
        mode=mode,
        include_text=True,
    )


@mcp.tool()
def bitrix_code_locator(
    query: str,
    limit: int = 15,
    project: str | None = None,
    lang: str | None = "php",
    path: str | None = None,
    php_namespace: str | None = None,
    php_class: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """Broad search for code locations without fetching full code text.

    BEST TOOL for exploration and navigation! Use this to find WHICH files and EXACTLY WHAT LINES 
    contain the relevant logic, classes, or functions. Highly token-efficient.
    It returns absolute file paths, line numbers, and symbol metadata WITHOUT the massive code body.
    Once you find the relevant lines, use your file reading tools to view the code.
    
    Use:
    - mode="qdrant-hybrid" (default) for concepts.
    - mode="qdrant-sparse" for exact symbols (e.g., 'BX.ajax', 'CUser::Add').
    
    Optional Filters:
    - limit: Max number of results (default 15).
    - lang: Filter by programming language (default "php", e.g., "javascript", "vue", "markdown").
    - path: Filter by file path substring (e.g., "local/components", "modules/sale").
    - project: Filter by project name.
    - php_namespace: Filter by exact PHP namespace (e.g., "Bitrix\\Sale").
    - php_class: Filter by exact PHP class or interface name (e.g., "Basket").
    """
    service = app_state.require_search_service()
    # Fetch with include_text=True so we can extract the signature, but we will strip the full text.
    response = service.search(
        query=query,
        limit=limit,
        project=project,
        lang=lang,
        path=path,
        php_namespace=php_namespace,
        php_class=php_class,
        mode=mode,
        include_text=True,
    )
    
    # Process results to remove large text payloads and extract signatures
    for result in response.get("results", []):
        text = result.pop("text", "")
        # Extract the first non-empty line as the signature
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        result["signature"] = lines[0] if lines else ""
        
        # Build a helpful symbol summary
        symbol_parts = []
        if result.get("php_nearest_type_name"):
            symbol_parts.append(f"{result.get('php_nearest_type_kind', 'class')} {result['php_nearest_type_name']}")
        if result.get("php_nearest_function_name"):
            symbol_parts.append(f"{result.get('php_nearest_function_kind', 'function')} {result['php_nearest_function_name']}")
        
        result["symbol"] = " :: ".join(symbol_parts) if symbol_parts else "file_chunk"

    return response


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
