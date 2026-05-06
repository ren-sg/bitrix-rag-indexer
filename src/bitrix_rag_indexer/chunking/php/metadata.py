from pathlib import Path
from typing import Any

from bitrix_rag_indexer.chunking.php.bitrix import build_bitrix_component_context_lines
from bitrix_rag_indexer.chunking.php.context import find_nearest_symbol_before
from bitrix_rag_indexer.chunking.php.models import (
    PhpContext,
    PhpDocConfig,
    PhpPayloadConfig,
    PhpPrefixConfig,
)
from bitrix_rag_indexer.chunking.php.phpdoc import PHPDOC_BLOCK_RE, parse_phpdoc_block
from bitrix_rag_indexer.parsing.tree_sitter_php import PhpAstSymbol


def build_php_prefix(
    path: Path,
    language: str,
    start_line: int,
    end_line: int,
    context: PhpContext,
    max_uses: int,
    prefix_config: PhpPrefixConfig,
) -> str:
    lines: list[str] = [
        f"Path: {path.as_posix()}",
        f"Language: {language}",
        f"Lines: {start_line}-{end_line}",
    ]

    if prefix_config.include_component_context:
        lines.extend(build_bitrix_component_context_lines(path))

    if context.namespace:
        lines.append(f"Namespace: {context.namespace}")

    append_php_uses_lines(
        lines=lines,
        uses=context.uses,
        max_uses=max_uses,
        prefix_config=prefix_config,
    )

    nearest_class = find_nearest_symbol_before(
        symbols=context.symbols,
        kinds={"class", "interface", "trait", "enum"},
        line=start_line,
    )
    if nearest_class:
        lines.append(f"Nearest type: {nearest_class.kind} {nearest_class.name}")

    nearest_function = find_nearest_symbol_before(
        symbols=context.symbols,
        kinds={"function", "method"},
        line=start_line,
    )
    if nearest_function:
        lines.append(
            f"Nearest function: {nearest_function.kind} {nearest_function.name}"
        )

    symbols_in_chunk = [
        symbol for symbol in context.symbols if start_line <= symbol.line <= end_line
    ]
    if symbols_in_chunk:
        lines.append("Symbols in chunk:")
        lines.extend(
            f"- {symbol.kind} {symbol.name} at line {symbol.line}"
            for symbol in symbols_in_chunk
        )

    return "\n".join(lines)

def build_php_symbol_prefix(
    path: Path,
    language: str,
    start_line: int,
    end_line: int,
    context: PhpContext,
    symbol: PhpAstSymbol,
    max_uses: int,
    prefix_config: PhpPrefixConfig,
) -> str:
    lines: list[str] = [
        f"Path: {path.as_posix()}",
        f"Language: {language}",
        f"Lines: {start_line}-{end_line}",
    ]

    if prefix_config.include_component_context:
        lines.extend(build_bitrix_component_context_lines(path))

    if context.namespace:
        lines.append(f"Namespace: {context.namespace}")

    append_php_uses_lines(
        lines=lines,
        uses=context.uses,
        max_uses=max_uses,
        prefix_config=prefix_config,
    )

    if symbol.parent_kind and symbol.parent_name:
        lines.append(f"Parent type: {symbol.parent_kind} {symbol.parent_name}")

    lines.append(
        build_php_symbol_label(
            symbol=symbol,
            prefix_config=prefix_config,
        )
    )

    if prefix_config.include_symbol_fqn:
        symbol_fqn = build_php_symbol_fqn(
            namespace=context.namespace,
            symbol=symbol,
        )
        if symbol_fqn:
            lines.append(f"Symbol FQN: {symbol_fqn}")

    return "\n".join(lines)

def build_php_symbol_metadata(
    context: PhpContext,
    symbol: PhpAstSymbol,
    max_uses: int,
) -> dict:
    return {
        "php_namespace": context.namespace,
        "php_uses": context.uses[:max_uses],
        "php_nearest_type_kind": symbol.parent_kind,
        "php_nearest_type_name": symbol.parent_name,
        "php_nearest_function_kind": symbol.kind,
        "php_nearest_function_name": symbol.name,
        "php_symbol_kind": symbol.kind,
        "php_symbol_name": symbol.name,
        "php_symbol_names": [symbol.name],
        "php_symbol_kinds": [symbol.kind],
        "php_symbols": [
            {
                "kind": symbol.kind,
                "name": symbol.name,
                "line": symbol.start_line,
                "parent_kind": symbol.parent_kind,
                "parent_name": symbol.parent_name,
                "visibility": symbol.visibility,
                "is_static": symbol.is_static,
                "is_abstract": symbol.is_abstract,
                "is_final": symbol.is_final,
                "has_body": symbol.has_body,
            }
        ],
        "php_symbol_visibility": symbol.visibility,
        "php_symbol_is_static": symbol.is_static,
        "php_symbol_is_abstract": symbol.is_abstract,
        "php_symbol_is_final": symbol.is_final,
        "php_symbol_has_body": symbol.has_body,
    }

def build_php_residual_metadata(
    context: PhpContext,
    start_line: int,
    end_line: int,
    max_uses: int,
) -> dict:
    symbols_in_chunk = [
        symbol
        for symbol in context.symbols
        if start_line <= symbol.line <= end_line
    ]

    nearest_type = find_nearest_symbol_before(
        symbols=context.symbols,
        kinds={"class", "interface", "trait", "enum"},
        line=start_line,
    )
    nearest_function = find_nearest_symbol_before(
        symbols=context.symbols,
        kinds={"function", "method"},
        line=start_line,
    )

    return {
        "php_namespace": context.namespace,
        "php_uses": context.uses[:max_uses],
        "php_nearest_type_kind": nearest_type.kind if nearest_type else None,
        "php_nearest_type_name": nearest_type.name if nearest_type else None,
        "php_nearest_function_kind": nearest_function.kind if nearest_function else None,
        "php_nearest_function_name": nearest_function.name if nearest_function else None,
        "php_symbol_names": [symbol.name for symbol in symbols_in_chunk],
        "php_symbol_kinds": [symbol.kind for symbol in symbols_in_chunk],
        "php_symbols": [
            {
                "kind": symbol.kind,
                "name": symbol.name,
                "line": symbol.line,
            }
            for symbol in symbols_in_chunk
        ],
    }

def build_phpdoc_config(raw_config: Any) -> PhpDocConfig:
    if not isinstance(raw_config, dict):
        return PhpDocConfig()

    include_tags = raw_config.get("include_tags", ("deprecated",))
    if not isinstance(include_tags, list | tuple):
        include_tags = ("deprecated",)

    normalized_tags = tuple(
        str(tag).strip().lstrip("@").casefold()
        for tag in include_tags
        if str(tag).strip()
    )

    return PhpDocConfig(
        enabled=bool(raw_config.get("enabled", True)),
        include_description=bool(raw_config.get("include_description", True)),
        include_tags=normalized_tags,
        max_chars=int(raw_config.get("max_chars", 1200)),
    )

def build_phpdoc_metadata(
    text: str,
    config: PhpDocConfig,
) -> dict[str, Any]:
    if not config.enabled:
        return {}

    infos = [
        parse_phpdoc_block(match.group(0))
        for match in PHPDOC_BLOCK_RE.finditer(text)
    ]
    if not infos:
        return {}

    tag_names: set[str] = set()
    summaries: list[str] = []

    for info in infos:
        if info.description:
            summaries.append(info.description)
        tag_names.update(info.tags)

    summary = summaries[0] if summaries else None
    if summary and len(summary) > 500:
        summary = summary[:500].rstrip() + "..."

    return {
        "php_doc_summary": summary,
        "php_doc_tags": sorted(tag_names),
        "php_doc_has_deprecated": "deprecated" in tag_names,
        "php_doc_has_param": "param" in tag_names,
        "php_doc_has_return": "return" in tag_names,
        "php_doc_has_throws": "throws" in tag_names,
    }

def build_php_prefix_config(raw_config: Any) -> PhpPrefixConfig:
    if not isinstance(raw_config, dict):
        return PhpPrefixConfig()

    return PhpPrefixConfig(
        include_uses=bool(raw_config.get("include_uses", True)),
        include_component_context=bool(
            raw_config.get("include_component_context", True)
        ),
        include_symbol_fqn=bool(raw_config.get("include_symbol_fqn", True)),
        include_symbol_modifiers=bool(
            raw_config.get("include_symbol_modifiers", True)
        ),
    )

def build_php_payload_config(raw_config: Any) -> PhpPayloadConfig:
    if not isinstance(raw_config, dict):
        return PhpPayloadConfig()

    return PhpPayloadConfig(
        include_uses=bool(raw_config.get("include_uses", True)),
    )

def apply_php_payload_config(
    *,
    metadata: dict[str, Any],
    payload_config: PhpPayloadConfig,
) -> None:
    if not payload_config.include_uses:
        metadata.pop("php_uses", None)

def append_php_uses_lines(
    lines: list[str],
    uses: list[str],
    max_uses: int,
    prefix_config: PhpPrefixConfig,
) -> None:
    if not prefix_config.include_uses or not uses:
        return

    selected_uses = uses[:max_uses]
    lines.append("Uses:")
    lines.extend(f"- {item}" for item in selected_uses)

def build_php_symbol_label(
    symbol: PhpAstSymbol,
    prefix_config: PhpPrefixConfig,
) -> str:
    symbol_name = symbol.name
    if symbol.parent_name and symbol.kind == "method":
        symbol_name = f"{symbol.parent_name}::{symbol.name}"

    parts: list[str] = ["Symbol:"]

    if prefix_config.include_symbol_modifiers:
        if symbol.visibility:
            parts.append(symbol.visibility)
        if symbol.is_static:
            parts.append("static")
        if symbol.is_abstract:
            parts.append("abstract")
        if symbol.is_final:
            parts.append("final")
        if not symbol.has_body:
            parts.append("declaration")

    parts.extend([symbol.kind, symbol_name])

    return " ".join(parts)

def build_php_symbol_fqn(
    namespace: str | None,
    symbol: PhpAstSymbol,
) -> str | None:
    if symbol.kind == "method" and symbol.parent_name:
        type_name = symbol.parent_name
        if namespace:
            type_name = f"{namespace}\\{type_name}"

        return f"{type_name}::{symbol.name}"

    if symbol.kind == "function":
        if namespace:
            return f"{namespace}\\{symbol.name}"

        return symbol.name

    return None

