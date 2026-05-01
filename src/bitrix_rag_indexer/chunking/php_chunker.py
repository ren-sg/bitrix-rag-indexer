from dataclasses import dataclass
from pathlib import Path
import re

from bitrix_rag_indexer.chunking.text_chunker import (
    TextChunk,
    split_by_lines_safely,
)
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
