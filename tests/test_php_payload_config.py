from pathlib import Path

from bitrix_rag_indexer.chunking.php_chunker import chunk_php


def test_php_payload_config_can_remove_uses_from_embedding_text_and_metadata() -> None:
    text = """<?php

namespace App\\Demo;

use Bitrix\\Main\\Loader;
use App\\Repository\\AreaRepository;

final class ExampleService
{
    public function getRows(): array
    {
        return AreaRepository::getList();
    }
}
"""

    chunks = chunk_php(
        text=text,
        path=Path("lib/ExampleService.php"),
        language="php",
        config={
            "strategy": "line",
            "max_chars": 4000,
            "overlap_chars": 0,
            "context": {
                "include_uses": False,
                "include_component_context": True,
                "include_symbol_fqn": True,
                "include_symbol_modifiers": True,
            },
            "payload": {
                "include_uses": False,
            },
        },
    )

    assert len(chunks) == 1

    chunk = chunks[0]

    assert "Uses:" not in chunk.text_for_embedding
    assert "Bitrix\\Main\\Loader" not in chunk.text_for_embedding
    assert "App\\Repository\\AreaRepository" in chunk.text
    assert "php_uses" not in chunk.metadata
