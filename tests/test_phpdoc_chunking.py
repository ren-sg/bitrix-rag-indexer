from pathlib import Path

from bitrix_rag_indexer.chunking.php_chunker import chunk_php


def test_phpdoc_embedding_keeps_description_and_deprecated_only() -> None:
    source = """<?php

namespace App\\Sizing;

/**
 * Сервис заявок sizing.
 */
class RequestService
{
    /**
     * Удаляет заявку согласования.
     *
     * Дополнительное смысловое описание метода.
     *
     * @param int $id Идентификатор заявки
     * @return bool
     * @throws \\RuntimeException
     * @deprecated use removeNew instead
     */
    public function remove(int $id): bool
    {
        return true;
    }
}
"""

    chunks = chunk_php(
        text=source,
        path=Path("php_interface/src/Sizing/RequestService.php"),
        language="php",
        config={
            "strategy": "tree-sitter",
            "max_chars": 2600,
            "overlap_chars": 300,
            "max_uses_in_prefix": 24,
            "fallback_strategy": "line",
            "phpdoc": {
                "enabled": True,
                "include_description": True,
                "include_tags": ["deprecated"],
                "max_chars": 1200,
            },
        },
    )

    method_chunk = next(
        chunk
        for chunk in chunks
        if chunk.metadata.get("php_symbol_name") == "remove"
    )

    assert "Удаляет заявку согласования." in method_chunk.text_for_embedding
    assert "Дополнительное смысловое описание метода." in method_chunk.text_for_embedding
    assert "@deprecated use removeNew instead" in method_chunk.text_for_embedding

    assert "@param" not in method_chunk.text_for_embedding
    assert "@return" not in method_chunk.text_for_embedding
    assert "@throws" not in method_chunk.text_for_embedding

    assert "@param int $id" in method_chunk.text
    assert "@return bool" in method_chunk.text
    assert "@throws \\RuntimeException" in method_chunk.text

    assert method_chunk.metadata["php_doc_has_deprecated"] is True
    assert method_chunk.metadata["php_doc_has_param"] is True
    assert method_chunk.metadata["php_doc_has_return"] is True
    assert method_chunk.metadata["php_doc_has_throws"] is True
    assert method_chunk.metadata["php_doc_tags"] == [
        "deprecated",
        "param",
        "return",
        "throws",
    ]


def test_phpdoc_can_be_disabled_for_embedding() -> None:
    source = """<?php

class Example
{
    /**
     * Важное описание метода.
     *
     * @deprecated old method
     */
    public function oldMethod(): void
    {
    }
}
"""

    chunks = chunk_php(
        text=source,
        path=Path("php_interface/src/Example.php"),
        language="php",
        config={
            "strategy": "tree-sitter",
            "max_chars": 2600,
            "overlap_chars": 300,
            "fallback_strategy": "line",
            "phpdoc": {
                "enabled": False,
            },
        },
    )

    method_chunk = next(
        chunk
        for chunk in chunks
        if chunk.metadata.get("php_symbol_name") == "oldMethod"
    )

    assert "Важное описание метода." not in method_chunk.text_for_embedding
    assert "@deprecated old method" not in method_chunk.text_for_embedding
    assert "public function oldMethod" in method_chunk.text_for_embedding

    assert "Важное описание метода." in method_chunk.text
    assert "@deprecated old method" in method_chunk.text
    assert "php_doc_summary" not in method_chunk.metadata
