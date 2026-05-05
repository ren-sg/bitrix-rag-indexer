from pathlib import Path

from bitrix_rag_indexer.chunking.php.context import extract_php_context, find_nearest_symbol_before
from bitrix_rag_indexer.chunking.php.metadata import (
    apply_php_payload_config,
    build_php_payload_config,
    build_php_prefix,
    build_php_prefix_config,
    build_php_residual_metadata,
    build_php_symbol_metadata,
    build_php_symbol_prefix,
    build_phpdoc_config,
    build_phpdoc_metadata,
)
from bitrix_rag_indexer.chunking.php.phpdoc import build_php_embedding_body
from bitrix_rag_indexer.chunking.php.residuals import (
    expand_start_line_for_docblock,
    find_residual_ranges,
    is_useful_residual_php_text,
    slice_lines,
    split_symbol_text_if_needed,
)
from bitrix_rag_indexer.chunking.text_chunker import TextChunk, split_by_lines_safely
from bitrix_rag_indexer.parsing.tree_sitter_php import parse_php_symbols
from bitrix_rag_indexer.state.hashes import stable_chunk_id


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
    payload_config = build_php_payload_config(config.get("payload"))

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

        apply_php_payload_config(
            metadata=metadata,
            payload_config=payload_config,
        )
        phpdoc_metadata = build_phpdoc_metadata(
            text=chunk_text_value,
            config=phpdoc_config,
        )
        metadata.update(phpdoc_metadata)

        embedding_body = build_php_embedding_body(
            text=chunk_text_value,
            phpdoc_config=phpdoc_config,
            payload_config=payload_config,
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
    payload_config = build_php_payload_config(config.get("payload"))

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

        apply_php_payload_config(
            metadata=metadata,
            payload_config=payload_config,
        )

        metadata["php_chunk_strategy"] = "tree-sitter"

        phpdoc_metadata = build_phpdoc_metadata(
            text=chunk_text_value,
            config=phpdoc_config,
        )
        metadata.update(phpdoc_metadata)

        embedding_body = build_php_embedding_body(
            text=chunk_text_value,
            phpdoc_config=phpdoc_config,
            payload_config=payload_config,
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

