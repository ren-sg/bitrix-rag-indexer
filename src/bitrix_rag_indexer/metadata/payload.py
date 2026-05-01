from pathlib import Path
from typing import Any

from bitrix_rag_indexer.state.hashes import sha256_text


def build_payload(
    source: dict[str, Any],
    file_path: Path,
    chunk: Any,
    language: str,
) -> dict[str, Any]:
    root = Path(source["root"]).resolve()
    rel_path = file_path.resolve().relative_to(root).as_posix()

    metadata = source.get("metadata", {})

    return {
        "source_name": source["name"],
        "source_type": source["type"],
        "source": metadata.get("source", source["name"]),
        "area": metadata.get("area"),
        "language": language,
        "path": file_path.as_posix(),
        "rel_path": rel_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "ordinal": chunk.ordinal,
        "content_hash": sha256_text(chunk.text_for_embedding),
        "text": chunk.text,
    }
