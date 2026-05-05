

import re

from bitrix_rag_indexer.chunking.php.models import PhpDocConfig, PhpDocInfo, PhpPayloadConfig

PHP_TOP_LEVEL_USE_IMPORT_RE = re.compile(
    r"^use\s+(?:function\s+|const\s+)?[^;]+;\s*$",
)

PHPDOC_BLOCK_RE = re.compile(r"/\*\*.*?\*/", re.DOTALL)

PHPDOC_TAG_RE = re.compile(r"^@([A-Za-z0-9_-]+)\b\s*(.*)$")

def build_php_embedding_body(
    *,
    text: str,
    phpdoc_config: PhpDocConfig,
    payload_config: PhpPayloadConfig,
) -> str:
    embedding_body = build_phpdoc_aware_text_for_embedding(
        text=text,
        config=phpdoc_config,
    )

    if not payload_config.include_uses:
        embedding_body = remove_php_top_level_use_imports(embedding_body)

    return embedding_body.strip()

def remove_php_top_level_use_imports(text: str) -> str:
    lines = text.splitlines()
    result: list[str] = []

    for line in lines:
        if PHP_TOP_LEVEL_USE_IMPORT_RE.match(line.strip()):
            continue

        result.append(line)

    return "\n".join(result).strip()

def build_phpdoc_aware_text_for_embedding(
    text: str,
    config: PhpDocConfig,
) -> str:
    if not PHPDOC_BLOCK_RE.search(text):
        return text

    def replace_docblock(match: re.Match[str]) -> str:
        if not config.enabled:
            return "\n"

        rendered = render_phpdoc_for_embedding(
            docblock=match.group(0),
            config=config,
        )
        if not rendered:
            return "\n"

        return f"\n{rendered}\n"

    return PHPDOC_BLOCK_RE.sub(replace_docblock, text).strip()

def render_phpdoc_for_embedding(
    docblock: str,
    config: PhpDocConfig,
) -> str:
    info = parse_phpdoc_block(docblock)
    lines: list[str] = []

    if config.include_description and info.description:
        lines.append("PHPDoc:")
        lines.append(info.description)

    for tag in config.include_tags:
        for value in info.tags.get(tag, []):
            lines.append(f"@{tag} {value}".rstrip())

    rendered = "\n".join(lines).strip()
    if config.max_chars > 0 and len(rendered) > config.max_chars:
        return rendered[: config.max_chars].rstrip() + "..."

    return rendered

def parse_phpdoc_block(docblock: str) -> PhpDocInfo:
    description_lines: list[str] = []
    tags: dict[str, list[str]] = {}
    seen_tag = False

    for raw_line in docblock.splitlines():
        line = clean_phpdoc_line(raw_line)
        if not line:
            continue

        tag_match = PHPDOC_TAG_RE.match(line)
        if tag_match:
            seen_tag = True
            tag_name = tag_match.group(1).casefold()
            tag_value = tag_match.group(2).strip()
            tags.setdefault(tag_name, []).append(tag_value)
            continue

        if not seen_tag:
            description_lines.append(line)

    return PhpDocInfo(
        description="\n".join(description_lines).strip(),
        tags=tags,
    )

def clean_phpdoc_line(line: str) -> str:
    stripped = line.strip()

    if stripped.startswith("/**"):
        stripped = stripped[3:].strip()
    if stripped.endswith("*/"):
        stripped = stripped[:-2].strip()
    if stripped.startswith("*"):
        stripped = stripped[1:].strip()

    return stripped

