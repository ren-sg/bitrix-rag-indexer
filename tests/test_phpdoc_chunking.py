from pathlib import Path

from bitrix_rag_indexer.chunking.php import chunk_php


def test_phpdoc_embedding_keeps_description_only() -> None:
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
    
    assert method_chunk.start_line == 20
    assert method_chunk.end_line == 23

    assert "Удаляет заявку согласования." in method_chunk.text_for_embedding
    assert "Дополнительное смысловое описание метода." in method_chunk.text_for_embedding
    assert "@deprecated use removeNew instead" not in method_chunk.text_for_embedding
    assert "Code:\npublic function remove" in method_chunk.text_for_embedding
    assert "PHPDoc Description:" in method_chunk.text_for_embedding

    assert "@param" not in method_chunk.text_for_embedding
    assert "@return" not in method_chunk.text_for_embedding
    assert "@throws" not in method_chunk.text_for_embedding

    assert "@param int $id" not in method_chunk.text
    assert "@return bool" not in method_chunk.text
    assert "@throws \\RuntimeException" not in method_chunk.text
    assert "Path: php_interface/src/Sizing/RequestService.php" not in method_chunk.text

    assert method_chunk.metadata["php_doc"]["has_deprecated"] is True
    assert method_chunk.metadata["php_doc"]["has_param"] is True
    assert method_chunk.metadata["php_doc"]["has_return"] is True
    assert method_chunk.metadata["php_doc"]["has_throws"] is True
    assert method_chunk.metadata["php_doc"]["tags"] == [
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
    
    assert method_chunk.start_line == 10
    assert method_chunk.end_line == 12

    assert "Важное описание метода." not in method_chunk.text_for_embedding
    assert "@deprecated old method" not in method_chunk.text_for_embedding
    assert "public function oldMethod" in method_chunk.text_for_embedding

    assert "Важное описание метода." not in method_chunk.text
    assert "@deprecated old method" not in method_chunk.text
    assert "php_doc" not in method_chunk.metadata
