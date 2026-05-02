from bitrix_rag_indexer.parsing.tree_sitter_php import parse_php_symbols


def test_parse_php_symbols_extracts_class_methods_and_function() -> None:
    text = """<?php

namespace App\\Iblock;

use Bitrix\\Main\\Loader;

class ElementRepository
{
    public function findById(int $id): array
    {
        return [];
    }

    protected static function mapRow(array $row): array
    {
        return $row;
    }
}

function helper_function(): void
{
}
"""

    symbols = parse_php_symbols(text)

    class_symbol = next(
        item for item in symbols if item.kind == "class" and item.name == "ElementRepository"
    )
    find_symbol = next(
        item for item in symbols if item.kind == "method" and item.name == "findById"
    )
    map_symbol = next(
        item for item in symbols if item.kind == "method" and item.name == "mapRow"
    )
    helper_symbol = next(
        item for item in symbols if item.kind == "function" and item.name == "helper_function"
    )

    assert class_symbol.parent_name is None

    assert find_symbol.parent_kind == "class"
    assert find_symbol.parent_name == "ElementRepository"
    assert find_symbol.start_line < find_symbol.end_line

    assert map_symbol.parent_kind == "class"
    assert map_symbol.parent_name == "ElementRepository"

    assert helper_symbol.parent_name is None


def test_parse_php_symbols_ignores_empty_text() -> None:
    assert parse_php_symbols("") == []


def test_parse_php_symbols_extracts_method_modifiers() -> None:
    text = """<?php

abstract class GridHandler
{
    abstract public function getRows(): array;

    final protected static function mapRow(array $row): array
    {
        return $row;
    }
}
"""

    symbols = parse_php_symbols(text)

    get_rows = next(item for item in symbols if item.name == "getRows")
    map_row = next(item for item in symbols if item.name == "mapRow")

    assert get_rows.visibility == "public"
    assert get_rows.is_abstract is True
    assert get_rows.has_body is False

    assert map_row.visibility == "protected"
    assert map_row.is_static is True
    assert map_row.is_final is True
    assert map_row.has_body is True
