from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import tree_sitter_php as tsphp
from tree_sitter import Language, Parser


@dataclass(frozen=True)
class PhpAstSymbol:
    kind: str
    name: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    parent_kind: str | None = None
    parent_name: str | None = None


NODE_KIND_MAP = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "trait_declaration": "trait",
    "enum_declaration": "enum",
    "method_declaration": "method",
    "function_definition": "function",
}

TYPE_KINDS = {"class", "interface", "trait", "enum"}
FUNCTION_KINDS = {"method", "function"}

NAME_NODE_TYPES = {
    "name",
    "identifier",
    "name_identifier",
    "qualified_name",
    "namespace_name",
}


@lru_cache(maxsize=1)
def get_php_parser() -> Parser:
    language = build_php_language()
    return build_parser(language)


def build_php_language() -> Language:
    if hasattr(tsphp, "language_php"):
        return Language(tsphp.language_php())

    if hasattr(tsphp, "language"):
        return Language(tsphp.language())

    raise RuntimeError(
        "tree_sitter_php does not expose language_php() or language(). "
        "Check installed tree-sitter-php version."
    )


def build_parser(language: Language) -> Parser:
    try:
        return Parser(language)
    except TypeError:
        parser = Parser()
        parser.language = language
        return parser


def parse_php_symbols(text: str) -> list[PhpAstSymbol]:
    if not text.strip():
        return []

    source_bytes = text.encode("utf-8", errors="replace")
    tree = get_php_parser().parse(source_bytes)

    symbols: list[PhpAstSymbol] = []

    def visit(
        node: Any,
        parent_kind: str | None = None,
        parent_name: str | None = None,
    ) -> None:
        mapped_kind = NODE_KIND_MAP.get(node.type)
        next_parent_kind = parent_kind
        next_parent_name = parent_name

        if mapped_kind:
            name = extract_node_name(source_bytes, node)

            if name:
                symbols.append(
                    PhpAstSymbol(
                        kind=mapped_kind,
                        name=name,
                        start_line=node_start_line(node),
                        end_line=node_end_line(node),
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        parent_kind=parent_kind if mapped_kind in FUNCTION_KINDS else None,
                        parent_name=parent_name if mapped_kind in FUNCTION_KINDS else None,
                    )
                )

                if mapped_kind in TYPE_KINDS:
                    next_parent_kind = mapped_kind
                    next_parent_name = name

        for child in node.named_children:
            visit(
                child,
                parent_kind=next_parent_kind,
                parent_name=next_parent_name,
            )

    visit(tree.root_node)

    return sorted(
        symbols,
        key=lambda item: (item.start_line, item.end_line, item.kind, item.name),
    )


def extract_node_name(source_bytes: bytes, node: Any) -> str | None:
    field_name_node = node.child_by_field_name("name")
    if field_name_node is not None:
        value = node_text(source_bytes, field_name_node).strip()
        if value:
            return value

    for child in node.named_children:
        if child.type in NAME_NODE_TYPES:
            value = node_text(source_bytes, child).strip()
            if value:
                return value

    return None


def node_text(source_bytes: bytes, node: Any) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode(
        "utf-8",
        errors="replace",
    )


def node_start_line(node: Any) -> int:
    return point_row(node.start_point) + 1


def node_end_line(node: Any) -> int:
    return point_row(node.end_point) + 1


def point_row(point: Any) -> int:
    if hasattr(point, "row"):
        return int(point.row)

    return int(point[0])
