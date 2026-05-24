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

_FILTER_DOCS = """
    Optional filters — omit any parameter you do not need.
    Do NOT pass null or empty strings; simply leave unused parameters out.

    - limit: Max number of results.
    - lang: Programming language filter (default "php", e.g. "javascript", "vue", "markdown").
    - project: Logical project name (e.g. "bitrix_modules"). Omit to search all projects.
    - path: Substring filter on stored rel_path in the Qdrant index.
      This matches the indexed payload, NOT necessarily the rel_path shown in results
      (display paths may include a project prefix such as "bitrix/modules/").
      For bitrix_modules, prefer "disk" or "bitrix/modules/disk", not "modules/disk".
      Start without path, then narrow down once you know the directory.
    - php_namespace: Exact PHP namespace (e.g. "Bitrix\\Sale").
    - php_class: Exact PHP class or interface name (e.g. "Basket").
    - mode: "qdrant-hybrid" (default), "qdrant-sparse", or "dense".

    Responses include applied_filters with only the filters that were actually used.
"""


@mcp.tool()
def bitrix_semantic_search(
    query: str,
    limit: int = 3,
    project: str = "",
    lang: str = "php",
    path: str = "",
    php_namespace: str = "",
    php_class: str = "",
    mode: str = "",
) -> dict[str, Any]:
    f"""Search for ACTUAL CODE implementations or semantic queries.

    BEST TOOL for fetching exact code snippets to solve problems.
    Returns the FULL code text, plus `rel_path` and (when `USE_ABS_PATH=true`) `abs_path`.
    IMPORTANT: Keep `limit` low (e.g., 2-4) to avoid flooding your context window!
    If you need to broadly explore many files or find where a method is used, use `bitrix_code_locator` instead.

    Use:
    - mode="qdrant-hybrid" (default) for semantic natural language queries.
    - mode="qdrant-sparse" for exact symbols (class names, exact function names).
    {_FILTER_DOCS}
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
        mode=mode or None,
        include_text=True,
    )


@mcp.tool()
def bitrix_code_locator(
    query: str,
    limit: int = 15,
    project: str = "",
    lang: str = "php",
    path: str = "",
    php_namespace: str = "",
    php_class: str = "",
    mode: str = "",
) -> dict[str, Any]:
    f"""Broad search for code locations without fetching full code text.

    BEST TOOL for exploration and navigation! Use this to find WHICH files and EXACTLY WHAT LINES
    contain the relevant logic, classes, or functions. Highly token-efficient.
    Returns `rel_path` (project-relative), and when `USE_ABS_PATH=true` also
    `abs_path` (absolute filesystem path), plus line numbers and symbol metadata WITHOUT the code body.
    Once you find the relevant lines, use your file reading tools to view the code.

    Use:
    - mode="qdrant-hybrid" (default) for concepts.
    - mode="qdrant-sparse" for exact symbols (e.g., 'BX.ajax', 'CUser::Add').
    {_FILTER_DOCS}
    """
    service = app_state.require_search_service()
    response = service.search(
        query=query,
        limit=limit,
        project=project,
        lang=lang,
        path=path,
        php_namespace=php_namespace,
        php_class=php_class,
        mode=mode or None,
        include_text=True,
    )

    for result in response.get("results", []):
        text = result.pop("text", "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        result["signature"] = lines[0] if lines else ""

        symbol_parts = []
        if result.get("php_nearest_type_name"):
            symbol_parts.append(
                f"{result.get('php_nearest_type_kind', 'class')} {result['php_nearest_type_name']}"
            )
        if result.get("php_nearest_function_name"):
            symbol_parts.append(
                f"{result.get('php_nearest_function_kind', 'function')} {result['php_nearest_function_name']}"
            )

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
