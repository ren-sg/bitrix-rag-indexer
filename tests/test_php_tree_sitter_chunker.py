from pathlib import Path

from bitrix_rag_indexer.chunking.php_chunker import chunk_php


def test_php_tree_sitter_chunker_creates_method_chunks() -> None:
    text = """<?php

namespace App\\Iblock;

use Bitrix\\Main\\Loader;

class ElementRepository
{
    /**
     * Find element by id.
     */
    public function findById(int $id): array
    {
        return [];
    }

    protected function mapRow(array $row): array
    {
        return $row;
    }
}
"""

    chunks = chunk_php(
        text=text,
        path=Path("local/php_interface/src/Iblock/ElementRepository.php"),
        language="php",
        config={
            "strategy": "tree-sitter",
            "max_chars": 2600,
            "overlap_chars": 300,
            "max_uses_in_prefix": 24,
        },
    )

    method_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("php_symbol_kind") == "method"
    ]

    assert [chunk.metadata["php_symbol_name"] for chunk in method_chunks] == [
        "findById",
        "mapRow",
    ]

    assert method_chunks[0].metadata["php_nearest_type_name"] == "ElementRepository"
    assert "Find element by id." in method_chunks[0].text
    # assert "Symbol: method findById" in method_chunks[0].text_for_embedding
    assert (
        "Symbol: public method ElementRepository::findById"
        in method_chunks[0].text_for_embedding
    )
    assert (
        "Symbol FQN: App\\Iblock\\ElementRepository::findById"
        in method_chunks[0].text_for_embedding
    )


def test_php_tree_sitter_chunker_falls_back_for_procedural_file() -> None:
    text = """<?php

use Bitrix\\Main\\Loader;

Loader::includeModule('iblock');

$result = CIBlockElement::GetList([]);
"""

    chunks = chunk_php(
        text=text,
        path=Path("local/php_interface/init.php"),
        language="php",
        config={
            "strategy": "tree-sitter",
            "max_chars": 2600,
            "overlap_chars": 300,
            "max_uses_in_prefix": 24,
            "fallback_strategy": "line",
        },
    )

    assert chunks
    assert "CIBlockElement::GetList" in chunks[0].text
