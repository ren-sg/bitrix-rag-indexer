from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from bitrix_rag_indexer.state.hashes import stable_chunk_id


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text: str
    text_for_embedding: str
    start_line: int
    end_line: int
    ordinal: int
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_text(
    text: str,
    path: Path,
    language: str,
    config: dict,
) -> list[TextChunk]:
    """
    Safe fallback chunker.

    Important:
    - line-based, not recursive;
    - no while loop with overlap recalculation;
    - cannot get stuck on the same offset;
    - handles huge single lines;
    - keeps line numbers.
    """
    max_chars = int(config.get("max_chars", 2200))
    overlap_chars = int(config.get("overlap_chars", 250))

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")

    if overlap_chars >= max_chars:
        overlap_chars = max_chars // 5

    if not text.strip():
        return []

    raw_chunks = split_by_lines_safely(
        text=text,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )

    chunks: list[TextChunk] = []

    for ordinal, raw in enumerate(raw_chunks, start=1):
        chunk_text_value = raw["text"].strip()

        if not chunk_text_value:
            continue

        start_line = raw["start_line"]
        end_line = raw["end_line"]

        prefix = (
            f"Path: {path.as_posix()}\n"
            f"Language: {language}\n"
            f"Lines: {start_line}-{end_line}\n\n"
        )

        text_for_embedding = prefix + chunk_text_value

        chunk_id = stable_chunk_id(
            path=path.as_posix(),
            ordinal=ordinal,
            text=text_for_embedding,
        )

        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                text=chunk_text_value,
                text_for_embedding=text_for_embedding,
                start_line=start_line,
                end_line=end_line,
                ordinal=ordinal,
            )
        )

    return chunks


def split_by_lines_safely(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[dict]:
    lines = text.splitlines(keepends=True)

    result: list[dict] = []

    current: list[tuple[int, str]] = []
    current_len = 0

    for line_no, line in enumerate(lines, start=1):
        # Huge generated/minified-like line.
        # Split it directly into fixed slices.
        if len(line) > max_chars:
            if current:
                result.append(make_raw_chunk(current))
                current = []
                current_len = 0

            for part in split_huge_line(line, max_chars=max_chars):
                result.append(
                    {
                        "text": part,
                        "start_line": line_no,
                        "end_line": line_no,
                    }
                )

            continue

        if current and current_len + len(line) > max_chars:
            result.append(make_raw_chunk(current))

            current = make_overlap_lines(
                lines=current,
                overlap_chars=overlap_chars,
            )
            current_len = sum(len(item[1]) for item in current)

        current.append((line_no, line))
        current_len += len(line)

    if current:
        result.append(make_raw_chunk(current))

    return result


def make_raw_chunk(lines: list[tuple[int, str]]) -> dict:
    return {
        "text": "".join(line for _, line in lines),
        "start_line": lines[0][0],
        "end_line": lines[-1][0],
    }


def make_overlap_lines(
    lines: list[tuple[int, str]],
    overlap_chars: int,
) -> list[tuple[int, str]]:
    if overlap_chars <= 0:
        return []

    selected: list[tuple[int, str]] = []
    total = 0

    for line_no, line in reversed(lines):
        if selected and total + len(line) > overlap_chars:
            break

        selected.append((line_no, line))
        total += len(line)

        if total >= overlap_chars:
            break

    selected.reverse()
    return selected


def split_huge_line(line: str, max_chars: int) -> list[str]:
    return [
        line[index : index + max_chars]
        for index in range(0, len(line), max_chars)
    ]
