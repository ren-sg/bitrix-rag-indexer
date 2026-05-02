from typing import Any

from rich.panel import Panel

DEBUG_PAYLOAD_FIELDS = [
    "php_namespace",
    "php_nearest_type_kind",
    "php_nearest_type_name",
    "php_nearest_function_kind",
    "php_nearest_function_name",
    "php_symbol_names",
    "php_symbol_kinds",
    "php_symbol_visibility",
    "php_symbol_is_static",
    "php_symbol_is_abstract",
    "php_symbol_is_final",
    "php_symbol_has_body",
    "module",
]


def format_search_result(item: dict[str, Any], debug: bool = False) -> Panel:
    payload = item.get("payload") or {}

    score = item.get("score", 0.0)
    source_name = payload.get("source_name", "?")
    language = payload.get("language", "?")
    rel_path = payload.get("rel_path") or item.get("path") or "?"
    start_line = payload.get("start_line", "?")
    end_line = payload.get("end_line", "?")

    title = (
        f"{_format_number(score)} | "
        f"{source_name} | "
        f"{language} | "
        f"{rel_path}:{start_line}-{end_line}"
    )

    text = item.get("text") or payload.get("text") or ""
    text = text[:1200]

    if debug:
        debug_text = format_debug_info(item)
        if debug_text:
            text = f"{debug_text}\n\n{text}"

    return Panel(
        text,
        title=title,
        expand=False,
    )


def format_debug_info(item: dict[str, Any]) -> str:
    lines: list[str] = []

    item_id = item.get("id")
    if item_id is not None:
        lines.append(f"id: {item_id}")

    hybrid_score = item.get("hybrid_score")
    if hybrid_score is not None:
        lines.append(f"hybrid_score: {_format_number(hybrid_score)}")

    dense_rank = item.get("dense_rank")
    dense_score = item.get("dense_score")
    if dense_rank is not None or dense_score is not None:
        lines.append(
            "dense: "
            f"rank={_format_optional(dense_rank)}, "
            f"score={_format_number(dense_score)}"
        )

    lexical_rank = item.get("lexical_rank")
    lexical_score = item.get("lexical_score")
    if lexical_rank is not None or lexical_score is not None:
        lines.append(
            "lexical: "
            f"rank={_format_optional(lexical_rank)}, "
            f"score={_format_number(lexical_score)}"
        )

    payload = item.get("payload") or {}
    for field_name in DEBUG_PAYLOAD_FIELDS:
        value = payload.get(field_name)
        if value in (None, "", [], {}):
            continue
        lines.append(f"{field_name}: {value}")

    return "\n".join(lines)


def _format_optional(value: Any) -> str:
    return "-" if value is None else str(value)


def _format_number(value: Any) -> str:
    if value is None:
        return "-"

    if isinstance(value, int | float):
        return f"{value:.6f}"

    return str(value)
