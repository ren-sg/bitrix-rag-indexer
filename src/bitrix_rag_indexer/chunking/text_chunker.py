from dataclasses import dataclass
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


def chunk_text(
    text: str,
    path: Path,
    language: str,
    config: dict,
) -> list[TextChunk]:
    max_chars = int(config.get("max_chars", 2200))
    overlap_chars = int(config.get("overlap_chars", 250))

    if not text.strip():
        return []

    line_offsets = build_line_offsets(text)
    parts = split_text_safely(
        text=text,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )

    chunks: list[TextChunk] = []

    for ordinal, part in enumerate(parts, start=1):
        start_index = part["start"]
        end_index = part["end"]
        chunk_text_value = part["text"].strip()

        if not chunk_text_value:
            continue

        start_line = line_number_for_offset(line_offsets, start_index)
        end_line = line_number_for_offset(line_offsets, end_index)

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


def split_text_safely(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[dict]:
    result: list[dict] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        target_end = min(start + max_chars, text_length)
        end = find_natural_break(text, start, target_end)

        if end <= start:
            end = target_end

        part = text[start:end]

        result.append(
            {
                "start": start,
                "end": end,
                "text": part,
            }
        )

        if end >= text_length:
            break

        start = max(0, end - overlap_chars)

    return result


def find_natural_break(text: str, start: int, target_end: int) -> int:
    window = text[start:target_end]

    separators = [
        "\n\n",
        "\nclass ",
        "\nfunction ",
        "\npublic function ",
        "\nprotected function ",
        "\nprivate function ",
        "\nexport ",
        "\nconst ",
        "\nlet ",
        "\nvar ",
    ]

    best = -1

    for separator in separators:
        pos = window.rfind(separator)
        if pos > best and pos > len(window) * 0.4:
            best = pos

    if best >= 0:
        return start + best

    newline_pos = window.rfind("\n")
    if newline_pos > len(window) * 0.6:
        return start + newline_pos

    return target_end


def build_line_offsets(text: str) -> list[int]:
    offsets = [0]
    for index, char in enumerate(text):
        if char == "\n":
            offsets.append(index + 1)
    return offsets


def line_number_for_offset(line_offsets: list[int], offset: int) -> int:
    line_number = 1

    for index, line_offset in enumerate(line_offsets, start=1):
        if line_offset > offset:
            break
        line_number = index

    return line_number
