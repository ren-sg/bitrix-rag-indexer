from dataclasses import dataclass
from pathlib import Path

from bitrix_rag_indexer.state.hashes import stable_chunk_id


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    text_for_embedding: str
    start_line: int
    end_line: int
    ordinal: int


def chunk_markdown(text: str, path: Path, config: dict) -> list[Chunk]:
    max_chars = int(config.get("max_chars", 1800))
    overlap_chars = int(config.get("overlap_chars", 200))

    lines = text.splitlines()
    blocks: list[tuple[int, int, str]] = []

    current_start = 1
    current: list[str] = []

    for i, line in enumerate(lines, start=1):
        starts_new_heading = line.startswith("# ") or line.startswith("## ")

        if starts_new_heading and current:
            blocks.append((current_start, i - 1, "\n".join(current).strip()))
            current_start = i
            current = [line]
        else:
            current.append(line)

    if current:
        blocks.append((current_start, len(lines), "\n".join(current).strip()))

    chunks: list[Chunk] = []
    ordinal = 0

    for start_line, end_line, block in blocks:
        if not block:
            continue

        parts = split_long_text(block, max_chars=max_chars, overlap_chars=overlap_chars)

        for part in parts:
            ordinal += 1
            prefix = f"Path: {path.as_posix()}\nLanguage: markdown\n\n"
            text_for_embedding = prefix + part

            chunk_id = stable_chunk_id(
                path=path.as_posix(),
                ordinal=ordinal,
                text=text_for_embedding,
            )

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=part,
                    text_for_embedding=text_for_embedding,
                    start_line=start_line,
                    end_line=end_line,
                    ordinal=ordinal,
                )
            )

    return chunks


def split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    result: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        part = text[start:end].strip()

        if part:
            result.append(part)

        if end >= len(text):
            break

        start = max(0, end - overlap_chars)

    return result
