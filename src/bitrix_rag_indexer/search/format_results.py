from typing import Any

from rich.panel import Panel


def format_search_result(item: dict[str, Any]) -> Panel:
    payload = item.get("payload") or {}

    score = item.get("score", 0.0)
    source_name = payload.get("source_name", "?")
    language = payload.get("language", "?")
    rel_path = payload.get("rel_path") or item.get("path") or "?"
    start_line = payload.get("start_line", "?")
    end_line = payload.get("end_line", "?")

    title = (
        f"{score:.4f} | "
        f"{source_name} | "
        f"{language} | "
        f"{rel_path}:{start_line}-{end_line}"
    )

    text = item.get("text") or payload.get("text") or ""
    text = text[:1200]

    return Panel(
        text,
        title=title,
        expand=False,
    )
