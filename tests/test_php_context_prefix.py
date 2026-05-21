from pathlib import Path

from bitrix_rag_indexer.chunking.php import chunk_php


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
    assert "Class: AreaRepository" in method_chunk.text_for_embedding
    assert (
        "Symbol: public static method AreaRepository::getList"
        in method_chunk.text_for_embedding
    )
    
    # Information removed from portable prefix but preserved in metadata
    assert "Uses:\n- Bitrix\\Main\\Loader" not in method_chunk.text_for_embedding
    assert "Symbol FQN:" not in method_chunk.text_for_embedding
    
    assert "App\\Sizing" == method_chunk.metadata["php_namespace"]
    assert "Bitrix\\Main\\Loader" in method_chunk.metadata["php_uses"]
    assert "public" == method_chunk.metadata["php_symbol_visibility"]
    assert method_chunk.metadata["php_symbol_is_static"] is True


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
        path=Path("components/vendor/crm.sizing.area.edit/class.php"),
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

    assert "vendor" == method_chunk.metadata["php_bitrix_vendor"]
    assert "crm.sizing.area.edit" == method_chunk.metadata["php_bitrix_component"]
    assert "components/vendor/crm.sizing.area.edit" == method_chunk.metadata["php_bitrix_path"]
    
    # Verify it is NOT in the embedding text (portable format)
    assert "Bitrix component:" not in method_chunk.text_for_embedding


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

    assert "bitrix" == function_chunk.metadata["php_bitrix_vendor"]
    assert "crm.kanban" == function_chunk.metadata["php_bitrix_component"]
    assert "bitrix24" == function_chunk.metadata["php_bitrix_site_template"]
    assert ".default" == function_chunk.metadata["php_bitrix_component_template"]
    
    # Verify it is NOT in the embedding text (portable format)
    assert "Bitrix component:" not in function_chunk.text_for_embedding
