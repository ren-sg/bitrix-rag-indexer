from pathlib import Path

from bitrix_rag_indexer.chunking.php_chunker import chunk_php


def test_php_symbol_prefix_contains_fqn_and_modifiers() -> None:
    source = """<?php

namespace App\\Sizing;

use Bitrix\\Main\\Loader;

class AreaRepository
{
    public static function getList(array $filter): array
    {
        return [];
    }
}
"""

    chunks = chunk_php(
        text=source,
        path=Path("php_interface/src/Sizing/AreaRepository.php"),
        language="php",
        config={
            "strategy": "tree-sitter",
            "max_chars": 2600,
            "overlap_chars": 300,
            "max_uses_in_prefix": 24,
            "fallback_strategy": "line",
            "context": {
                "include_uses": True,
                "include_component_context": True,
                "include_symbol_fqn": True,
                "include_symbol_modifiers": True,
            },
        },
    )

    method_chunk = next(
        chunk
        for chunk in chunks
        if chunk.metadata.get("php_symbol_name") == "getList"
    )

    assert "Namespace: App\\Sizing" in method_chunk.text_for_embedding
    assert "Uses:\n- Bitrix\\Main\\Loader" in method_chunk.text_for_embedding
    assert "Parent type: class AreaRepository" in method_chunk.text_for_embedding
    assert (
        "Symbol: public static method AreaRepository::getList"
        in method_chunk.text_for_embedding
    )
    assert (
        "Symbol FQN: App\\Sizing\\AreaRepository::getList"
        in method_chunk.text_for_embedding
    )


def test_php_component_prefix_contains_bitrix_component_context() -> None:
    source = """<?php

class CrmSizingAreaEditComponent extends CBitrixComponent
{
    public function executeComponent(): void
    {
    }
}
"""

    chunks = chunk_php(
        text=source,
        path=Path("components/nlmk/crm.sizing.area.edit/class.php"),
        language="php",
        config={
            "strategy": "tree-sitter",
            "max_chars": 2600,
            "overlap_chars": 300,
            "max_uses_in_prefix": 24,
            "fallback_strategy": "line",
            "context": {
                "include_uses": True,
                "include_component_context": True,
                "include_symbol_fqn": True,
                "include_symbol_modifiers": True,
            },
        },
    )

    method_chunk = next(
        chunk
        for chunk in chunks
        if chunk.metadata.get("php_symbol_name") == "executeComponent"
    )

    assert (
        "Bitrix component: nlmk:crm.sizing.area.edit"
        in method_chunk.text_for_embedding
    )
    assert (
        "Bitrix component path: components/nlmk/crm.sizing.area.edit"
        in method_chunk.text_for_embedding
    )


def test_php_template_prefix_contains_bitrix_component_context() -> None:
    source = """<?php

function renderKanban(): void
{
}
"""

    chunks = chunk_php(
        text=source,
        path=Path(
            "templates/bitrix24/components/bitrix/crm.kanban/.default/template.php"
        ),
        language="php",
        config={
            "strategy": "tree-sitter",
            "max_chars": 2600,
            "overlap_chars": 300,
            "max_uses_in_prefix": 24,
            "fallback_strategy": "line",
            "context": {
                "include_uses": True,
                "include_component_context": True,
                "include_symbol_fqn": True,
                "include_symbol_modifiers": True,
            },
        },
    )

    function_chunk = next(
        chunk
        for chunk in chunks
        if chunk.metadata.get("php_symbol_name") == "renderKanban"
    )

    assert "Bitrix component: bitrix:crm.kanban" in function_chunk.text_for_embedding
    assert "Bitrix site template: bitrix24" in function_chunk.text_for_embedding
    assert "Bitrix component template: .default" in function_chunk.text_for_embedding
