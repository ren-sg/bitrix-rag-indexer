

from bitrix_rag_indexer.chunking.text_chunker import split_by_lines_safely


def expand_start_line_for_docblock(
    lines: list[str],
    start_line: int,
) -> int:
    index = start_line - 2

    while index >= 0 and not lines[index].strip():
        index -= 1

    if index < 0:
        return start_line

    if not lines[index].strip().endswith("*/"):
        return start_line

    while index >= 0:
        stripped = lines[index].strip()
        if stripped.startswith("/**"):
            return index + 1
        index -= 1

    return start_line

def split_symbol_text_if_needed(
    text: str,
    start_line: int,
    max_chars: int,
    overlap_chars: int,
) -> list[dict]:
    if len(text) <= max_chars:
        line_count = max(1, len(text.splitlines()))
        return [
            {
                "text": text,
                "start_line": start_line,
                "end_line": start_line + line_count - 1,
            }
        ]

    parts = split_by_lines_safely(
        text=text,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )

    adjusted: list[dict] = []
    for part in parts:
        adjusted.append(
            {
                "text": part["text"],
                "start_line": start_line + int(part["start_line"]) - 1,
                "end_line": start_line + int(part["end_line"]) - 1,
            }
        )

    return adjusted

def slice_lines(
    lines: list[str],
    start_line: int,
    end_line: int,
) -> str:
    return "\n".join(lines[start_line - 1 : end_line])

def find_residual_ranges(
    total_lines: int,
    covered_ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if total_lines <= 0:
        return []

    if not covered_ranges:
        return [(1, total_lines)]

    merged = merge_line_ranges(covered_ranges)
    residual: list[tuple[int, int]] = []
    cursor = 1

    for start_line, end_line in merged:
        if cursor < start_line:
            residual.append((cursor, start_line - 1))
        cursor = max(cursor, end_line + 1)

    if cursor <= total_lines:
        residual.append((cursor, total_lines))

    return residual

def merge_line_ranges(
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    normalized = sorted(
        (min(start, end), max(start, end))
        for start, end in ranges
    )

    merged: list[tuple[int, int]] = []

    for start_line, end_line in normalized:
        if not merged:
            merged.append((start_line, end_line))
            continue

        last_start, last_end = merged[-1]
        if start_line <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end_line))
        else:
            merged.append((start_line, end_line))

    return merged

def is_useful_residual_php_text(text: str) -> bool:
    stripped = text.strip()

    if not stripped:
        return False

    useless = {
        "<?php",
        "?>",
        "{",
        "}",
        "};",
    }

    if stripped in useless:
        return False

    if all(line.strip() in useless for line in stripped.splitlines()):
        return False

    return True

