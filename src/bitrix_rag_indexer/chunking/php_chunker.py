import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bitrix_rag_indexer.chunking.text_chunker import (
    TextChunk,
    split_by_lines_safely,
)
from bitrix_rag_indexer.parsing.tree_sitter_php import PhpAstSymbol, parse_php_symbols
from bitrix_rag_indexer.state.hashes import stable_chunk_id


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


@dataclass(frozen=True)
class PhpDocConfig:
    enabled: bool = True
    include_description: bool = True
    include_tags: tuple[str, ...] = ("deprecated",)
    max_chars: int = 1200


@dataclass(frozen=True)
class PhpDocInfo:
    description: str
    tags: dict[str, list[str]]


@dataclass(frozen=True)
class PhpPrefixConfig:
    include_uses: bool = True
    include_component_context: bool = True
    include_symbol_fqn: bool = True
    include_symbol_modifiers: bool = True


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

PHPDOC_BLOCK_RE = re.compile(r"/\*\*.*?\*/", re.DOTALL)
PHPDOC_TAG_RE = re.compile(r"^@([A-Za-z0-9_-]+)\b\s*(.*)$")


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
    phpdoc_config = build_phpdoc_config(config.get("phpdoc"))
    prefix_config = build_php_prefix_config(config.get("context"))

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
            prefix_config=prefix_config,
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

        phpdoc_metadata = build_phpdoc_metadata(
            text=chunk_text_value,
            config=phpdoc_config,
        )
        metadata.update(phpdoc_metadata)

        embedding_body = build_phpdoc_aware_text_for_embedding(
            text=chunk_text_value,
            config=phpdoc_config,
        )

        text_for_embedding = prefix + "\n\n" + embedding_body
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
    phpdoc_config = build_phpdoc_config(config.get("phpdoc"))
    prefix_config = build_php_prefix_config(config.get("context"))

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
                prefix_config=prefix_config,
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
                prefix_config=prefix_config,
            )
            metadata = build_php_residual_metadata(
                context=context,
                start_line=start_line,
                end_line=end_line,
                max_uses=max_uses,
            )

        metadata["php_chunk_strategy"] = "tree-sitter"

        phpdoc_metadata = build_phpdoc_metadata(
            text=chunk_text_value,
            config=phpdoc_config,
        )
        metadata.update(phpdoc_metadata)

        embedding_body = build_phpdoc_aware_text_for_embedding(
            text=chunk_text_value,
            config=phpdoc_config,
        )
        text_for_embedding = prefix + "\n\n" + embedding_body
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


def build_phpdoc_aware_text_for_embedding(
    text: str,
    config: PhpDocConfig,
) -> str:
    if not PHPDOC_BLOCK_RE.search(text):
        return text

    def replace_docblock(match: re.Match[str]) -> str:
        if not config.enabled:
            return "\n"

        rendered = render_phpdoc_for_embedding(
            docblock=match.group(0),
            config=config,
        )
        if not rendered:
            return "\n"

        return f"\n{rendered}\n"

    return PHPDOC_BLOCK_RE.sub(replace_docblock, text).strip()


def render_phpdoc_for_embedding(
    docblock: str,
    config: PhpDocConfig,
) -> str:
    info = parse_phpdoc_block(docblock)
    lines: list[str] = []

    if config.include_description and info.description:
        lines.append("PHPDoc:")
        lines.append(info.description)

    for tag in config.include_tags:
        for value in info.tags.get(tag, []):
            lines.append(f"@{tag} {value}".rstrip())

    rendered = "\n".join(lines).strip()
    if config.max_chars > 0 and len(rendered) > config.max_chars:
        return rendered[: config.max_chars].rstrip() + "..."

    return rendered


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


def parse_phpdoc_block(docblock: str) -> PhpDocInfo:
    description_lines: list[str] = []
    tags: dict[str, list[str]] = {}
    seen_tag = False

    for raw_line in docblock.splitlines():
        line = clean_phpdoc_line(raw_line)
        if not line:
            continue

        tag_match = PHPDOC_TAG_RE.match(line)
        if tag_match:
            seen_tag = True
            tag_name = tag_match.group(1).casefold()
            tag_value = tag_match.group(2).strip()
            tags.setdefault(tag_name, []).append(tag_value)
            continue

        if not seen_tag:
            description_lines.append(line)

    return PhpDocInfo(
        description="\n".join(description_lines).strip(),
        tags=tags,
    )


def clean_phpdoc_line(line: str) -> str:
    stripped = line.strip()

    if stripped.startswith("/**"):
        stripped = stripped[3:].strip()
    if stripped.endswith("*/"):
        stripped = stripped[:-2].strip()
    if stripped.startswith("*"):
        stripped = stripped[1:].strip()

    return stripped


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


def build_bitrix_component_context_lines(path: Path) -> list[str]:
    component = detect_bitrix_component_context(path)
    if component is None:
        return []

    lines = [
        f"Bitrix component: {component['vendor']}:{component['name']}",
        f"Bitrix component path: {component['component_path']}",
    ]

    site_template = component.get("site_template")
    if site_template:
        lines.append(f"Bitrix site template: {site_template}")

    component_template = component.get("component_template")
    if component_template:
        lines.append(f"Bitrix component template: {component_template}")

    return lines


def detect_bitrix_component_context(path: Path) -> dict[str, str] | None:
    parts = path.as_posix().split("/")

    if len(parts) >= 3 and parts[0] == "components":
        return {
            "vendor": parts[1],
            "name": parts[2],
            "component_path": "/".join(parts[:3]),
        }

    if len(parts) >= 6 and parts[0] == "templates":
        try:
            components_index = parts.index("components")
        except ValueError:
            return None

        if len(parts) <= components_index + 2:
            return None

        component_template = (
            parts[components_index + 3]
            if len(parts) > components_index + 3
            else None
        )

        result = {
            "vendor": parts[components_index + 1],
            "name": parts[components_index + 2],
            "component_path": "/".join(parts[: components_index + 3]),
            "site_template": parts[1],
        }

        if component_template:
            result["component_template"] = component_template

        return result

    return None


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
