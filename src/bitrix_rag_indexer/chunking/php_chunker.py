from dataclasses import dataclass
from pathlib import Path
import re

from bitrix_rag_indexer.chunking.text_chunker import (
    TextChunk,
    split_by_lines_safely,
)
from bitrix_rag_indexer.state.hashes import stable_chunk_id
from bitrix_rag_indexer.parsing.tree_sitter_php import PhpAstSymbol, parse_php_symbols


@dataclass(frozen=True)
class PhpSymbol:
    kind: str
    name: str
    line: int


@dataclass(frozen=True)
class PhpContext:
    namespace: str | None
    uses: list[str]
    symbols: list[PhpSymbol]


NAMESPACE_RE = re.compile(
    r"^\s*namespace\s+([^;{]+)\s*[;{]",
)

USE_RE = re.compile(
    r"^\s*use\s+([^;]+);",
)

CLASS_RE = re.compile(
    r"^\s*(?:abstract\s+|final\s+)?"
    r"(class|interface|trait|enum)\s+"
    r"([A-Za-z_\x80-\xff][A-Za-z0-9_\x80-\xff]*)"
)

FUNCTION_RE = re.compile(
    r"^\s*"
    r"(?:(public|protected|private)\s+)?"
    r"(?:(?:static|final|abstract)\s+)*"
    r"function\s+&?\s*"
    r"([A-Za-z_\x80-\xff][A-Za-z0-9_\x80-\xff]*)\s*\("
)


def chunk_php(
    text: str,
    path: Path,
    language: str,
    config: dict,
) -> list[TextChunk]:
    strategy = str(config.get("strategy", "line")).lower()

    if strategy == "tree-sitter":
        try:
            chunks = chunk_php_tree_sitter(
                text=text,
                path=path,
                language=language,
                config=config,
            )
            if chunks:
                return chunks
        except Exception:
            if config.get("fallback_strategy", "line") != "line":
                raise

    return chunk_php_line_based(
        text=text,
        path=path,
        language=language,
        config=config,
    )


def chunk_php_line_based(
    text: str,
    path: Path,
    language: str,
    config: dict,
) -> list[TextChunk]:
    """
    Safe PHP-aware chunker.

    Still line-based and memory-safe, but enriches text_for_embedding
    with structural PHP context extracted from the same file.
    """

    max_chars = int(config.get("max_chars", 2600))
    overlap_chars = int(config.get("overlap_chars", 300))
    max_uses = int(config.get("max_uses_in_prefix", 24))

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")

    if overlap_chars >= max_chars:
        overlap_chars = max_chars // 5

    if not text.strip():
        return []

    context = extract_php_context(text)

    raw_chunks = split_by_lines_safely(
        text=text,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )

    chunks: list[TextChunk] = []

    for ordinal, raw in enumerate(raw_chunks, start=1):
        chunk_text_value = raw["text"].strip()

        if not chunk_text_value:
            continue

        start_line = int(raw["start_line"])
        end_line = int(raw["end_line"])

        prefix = build_php_prefix(
            path=path,
            language=language,
            start_line=start_line,
            end_line=end_line,
            context=context,
            max_uses=max_uses,
        )

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
        symbols_in_chunk = [
            symbol
            for symbol in context.symbols
            if start_line <= symbol.line <= end_line
        ]

        metadata = {
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

        text_for_embedding = prefix + "\n\n" + chunk_text_value
        chunk_id = stable_chunk_id(
            path=path.as_posix(),
            ordinal=ordinal,
            text=text_for_embedding,
        )

        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                text=chunk_text_value,
                text_for_embedding=text_for_embedding,
                start_line=start_line,
                end_line=end_line,
                ordinal=ordinal,
                metadata=metadata,
            )
        )

    return chunks


def chunk_php_tree_sitter(
    text: str,
    path: Path,
    language: str,
    config: dict,
) -> list[TextChunk]:
    max_chars = int(config.get("max_chars", 2600))
    overlap_chars = int(config.get("overlap_chars", 300))
    max_uses = int(config.get("max_uses_in_prefix", 24))

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")

    if overlap_chars >= max_chars:
        overlap_chars = max_chars // 5

    if not text.strip():
        return []

    context = extract_php_context(text)
    symbols = parse_php_symbols(text)

    callable_symbols = [
        symbol
        for symbol in symbols
        if symbol.kind in {"method", "function"}
    ]

    if not callable_symbols:
        return chunk_php_line_based(
            text=text,
            path=path,
            language=language,
            config=config,
        )

    lines = text.splitlines()
    covered_ranges: list[tuple[int, int]] = []
    chunk_specs: list[dict] = []

    for symbol in callable_symbols:
        start_line = expand_start_line_for_docblock(
            lines=lines,
            start_line=symbol.start_line,
        )
        end_line = symbol.end_line

        if start_line > end_line:
            continue

        symbol_text = slice_lines(
            lines=lines,
            start_line=start_line,
            end_line=end_line,
        ).strip()

        if not symbol_text:
            continue

        covered_ranges.append((start_line, end_line))

        for part in split_symbol_text_if_needed(
            text=symbol_text,
            start_line=start_line,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        ):
            chunk_specs.append(
                {
                    "kind": "symbol",
                    "text": part["text"],
                    "start_line": part["start_line"],
                    "end_line": part["end_line"],
                    "symbol": symbol,
                }
            )

    for residual_range in find_residual_ranges(
        total_lines=len(lines),
        covered_ranges=covered_ranges,
    ):
        residual_text = slice_lines(
            lines=lines,
            start_line=residual_range[0],
            end_line=residual_range[1],
        ).strip()

        if not is_useful_residual_php_text(residual_text):
            continue

        for part in split_symbol_text_if_needed(
            text=residual_text,
            start_line=residual_range[0],
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        ):
            chunk_specs.append(
                {
                    "kind": "residual",
                    "text": part["text"],
                    "start_line": part["start_line"],
                    "end_line": part["end_line"],
                    "symbol": None,
                }
            )

    chunk_specs.sort(key=lambda item: (item["start_line"], item["end_line"]))

    chunks: list[TextChunk] = []
    for ordinal, spec in enumerate(chunk_specs, start=1):
        chunk_text_value = spec["text"].strip()
        if not chunk_text_value:
            continue

        start_line = int(spec["start_line"])
        end_line = int(spec["end_line"])
        symbol = spec["symbol"]

        if symbol is not None:
            prefix = build_php_symbol_prefix(
                path=path,
                language=language,
                start_line=start_line,
                end_line=end_line,
                context=context,
                symbol=symbol,
                max_uses=max_uses,
            )
            metadata = build_php_symbol_metadata(
                context=context,
                symbol=symbol,
                max_uses=max_uses,
            )
        else:
            prefix = build_php_prefix(
                path=path,
                language=language,
                start_line=start_line,
                end_line=end_line,
                context=context,
                max_uses=max_uses,
            )
            metadata = build_php_residual_metadata(
                context=context,
                start_line=start_line,
                end_line=end_line,
                max_uses=max_uses,
            )

        metadata["php_chunk_strategy"] = "tree-sitter"

        text_for_embedding = prefix + "\n\n" + chunk_text_value
        chunk_id = stable_chunk_id(
            path=path.as_posix(),
            ordinal=ordinal,
            text=text_for_embedding,
        )

        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                text=chunk_text_value,
                text_for_embedding=text_for_embedding,
                start_line=start_line,
                end_line=end_line,
                ordinal=ordinal,
                metadata=metadata,
            )
        )

    return chunks


def extract_php_context(text: str) -> PhpContext:
    namespace: str | None = None
    uses: list[str] = []
    symbols: list[PhpSymbol] = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        if namespace is None:
            namespace_match = NAMESPACE_RE.match(line)
            if namespace_match:
                namespace = namespace_match.group(1).strip()

        use_match = USE_RE.match(line)
        if use_match:
            uses.append(use_match.group(1).strip())

        class_match = CLASS_RE.match(line)
        if class_match:
            symbols.append(
                PhpSymbol(
                    kind=class_match.group(1),
                    name=class_match.group(2),
                    line=line_no,
                )
            )

        function_match = FUNCTION_RE.match(line)
        if function_match:
            visibility = function_match.group(1)

            symbols.append(
                PhpSymbol(
                    kind="method" if visibility else "function",
                    name=function_match.group(2),
                    line=line_no,
                )
            )

    return PhpContext(
        namespace=namespace,
        uses=dedupe_keep_order(uses),
        symbols=symbols,
    )


def build_php_prefix(
    path: Path,
    language: str,
    start_line: int,
    end_line: int,
    context: PhpContext,
    max_uses: int,
) -> str:
    lines: list[str] = [
        f"Path: {path.as_posix()}",
        f"Language: {language}",
        f"Lines: {start_line}-{end_line}",
    ]

    if context.namespace:
        lines.append(f"Namespace: {context.namespace}")

    if context.uses:
        selected_uses = context.uses[:max_uses]
        lines.append("Uses:")
        lines.extend(f"- {item}" for item in selected_uses)

    nearest_class = find_nearest_symbol_before(
        symbols=context.symbols,
        kinds={"class", "interface", "trait", "enum"},
        line=start_line,
    )

    if nearest_class:
        lines.append(
            f"Nearest type: {nearest_class.kind} {nearest_class.name}"
        )

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
        symbol
        for symbol in context.symbols
        if start_line <= symbol.line <= end_line
    ]

    if symbols_in_chunk:
        lines.append("Symbols in chunk:")
        lines.extend(
            f"- {symbol.kind} {symbol.name} at line {symbol.line}"
            for symbol in symbols_in_chunk
        )

    return "\n".join(lines)


def find_nearest_symbol_before(
    symbols: list[PhpSymbol],
    kinds: set[str],
    line: int,
) -> PhpSymbol | None:
    candidates = [
        symbol
        for symbol in symbols
        if symbol.kind in kinds and symbol.line <= line
    ]

    if not candidates:
        return None

    return candidates[-1]


def dedupe_keep_order(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for item in items:
        if item in seen:
            continue

        result.append(item)
        seen.add(item)

    return result

def build_php_symbol_prefix(
    path: Path,
    language: str,
    start_line: int,
    end_line: int,
    context: PhpContext,
    symbol: PhpAstSymbol,
    max_uses: int,
) -> str:
    lines: list[str] = [
        f"Path: {path.as_posix()}",
        f"Language: {language}",
        f"Lines: {start_line}-{end_line}",
    ]

    if context.namespace:
        lines.append(f"Namespace: {context.namespace}")

    if context.uses:
        selected_uses = context.uses[:max_uses]
        lines.append("Uses:")
        lines.extend(f"- {item}" for item in selected_uses)

    if symbol.parent_kind and symbol.parent_name:
        lines.append(f"Parent type: {symbol.parent_kind} {symbol.parent_name}")

    lines.append(f"Symbol: {symbol.kind} {symbol.name}")

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


def expand_start_line_for_docblock(
    lines: list[str],
    start_line: int,
) -> int:
    index = start_line - 2

    while index >= 0 and not lines[index].strip():
        index -= 1

    if index < 0:
        return start_line

    if not lines[index].strip().endswith("*/"):
        return start_line

    while index >= 0:
        stripped = lines[index].strip()
        if stripped.startswith("/**"):
            return index + 1
        index -= 1

    return start_line


def split_symbol_text_if_needed(
    text: str,
    start_line: int,
    max_chars: int,
    overlap_chars: int,
) -> list[dict]:
    if len(text) <= max_chars:
        line_count = max(1, len(text.splitlines()))
        return [
            {
                "text": text,
                "start_line": start_line,
                "end_line": start_line + line_count - 1,
            }
        ]

    parts = split_by_lines_safely(
        text=text,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )

    adjusted: list[dict] = []
    for part in parts:
        adjusted.append(
            {
                "text": part["text"],
                "start_line": start_line + int(part["start_line"]) - 1,
                "end_line": start_line + int(part["end_line"]) - 1,
            }
        )

    return adjusted


def slice_lines(
    lines: list[str],
    start_line: int,
    end_line: int,
) -> str:
    return "\n".join(lines[start_line - 1 : end_line])


def find_residual_ranges(
    total_lines: int,
    covered_ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if total_lines <= 0:
        return []

    if not covered_ranges:
        return [(1, total_lines)]

    merged = merge_line_ranges(covered_ranges)
    residual: list[tuple[int, int]] = []
    cursor = 1

    for start_line, end_line in merged:
        if cursor < start_line:
            residual.append((cursor, start_line - 1))
        cursor = max(cursor, end_line + 1)

    if cursor <= total_lines:
        residual.append((cursor, total_lines))

    return residual


def merge_line_ranges(
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    normalized = sorted(
        (min(start, end), max(start, end))
        for start, end in ranges
    )

    merged: list[tuple[int, int]] = []

    for start_line, end_line in normalized:
        if not merged:
            merged.append((start_line, end_line))
            continue

        last_start, last_end = merged[-1]
        if start_line <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end_line))
        else:
            merged.append((start_line, end_line))

    return merged


def is_useful_residual_php_text(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return False

    useless = {
        "<?php",
        "?>",
        "{",
        "}",
        "};",
    }

    if stripped in useless:
        return False

    if all(line.strip() in useless for line in stripped.splitlines()):
        return False

    return True
