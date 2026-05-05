from pathlib import Path

from bitrix_rag_indexer.chunking.php.strategies import chunk_php_line_based, chunk_php_tree_sitter
from bitrix_rag_indexer.chunking.text_chunker import TextChunk


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
            pass

    return chunk_php_line_based(
        text=text,
        path=path,
        language=language,
        config=config,
    )
