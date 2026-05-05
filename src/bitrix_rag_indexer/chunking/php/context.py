import re

from bitrix_rag_indexer.chunking.php.metadata import dedupe_keep_order
from bitrix_rag_indexer.chunking.php.models import PhpContext, PhpSymbol

NAMESPACE_RE = re.compile(
    r"^\s*namespace\s+([^;{]+)\s*[;{]",
)

USE_RE = re.compile(
    r"^\s*use\s+([^;]+);",
)

CLASS_RE = re.compile(
    r"^\s*(?:abstract\s+|final\s+)?"
    r"(class|interface|trait|enum)\s+"
    r"([A-Za-z_\x80-\xff][A-Za-z0-9_\x80-\xff]*)"
)

FUNCTION_RE = re.compile(
    r"^\s*"
    r"(?:(public|protected|private)\s+)?"
    r"(?:(?:static|final|abstract)\s+)*"
    r"function\s+&?\s*"
    r"([A-Za-z_\x80-\xff][A-Za-z0-9_\x80-\xff]*)\s*\("
)

def extract_php_context(text: str) -> PhpContext:
    namespace: str | None = None
    uses: list[str] = []
    symbols: list[PhpSymbol] = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        if namespace is None:
            namespace_match = NAMESPACE_RE.match(line)
            if namespace_match:
                namespace = namespace_match.group(1).strip()

        use_match = USE_RE.match(line)
        if use_match:
            uses.append(use_match.group(1).strip())

        class_match = CLASS_RE.match(line)
        if class_match:
            symbols.append(
                PhpSymbol(
                    kind=class_match.group(1),
                    name=class_match.group(2),
                    line=line_no,
                )
            )

        function_match = FUNCTION_RE.match(line)
        if function_match:
            visibility = function_match.group(1)

            symbols.append(
                PhpSymbol(
                    kind="method" if visibility else "function",
                    name=function_match.group(2),
                    line=line_no,
                )
            )

    return PhpContext(
        namespace=namespace,
        uses=dedupe_keep_order(uses),
        symbols=symbols,
    )

def find_nearest_symbol_before(
    symbols: list[PhpSymbol],
    kinds: set[str],
    line: int,
) -> PhpSymbol | None:
    candidates = [
        symbol
        for symbol in symbols
        if symbol.kind in kinds and symbol.line <= line
    ]

    if not candidates:
        return None

    return candidates[-1]

