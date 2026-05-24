from pathlib import Path
from typing import Any

from bitrix_rag_indexer.config.project import ProjectConfig, compute_rel_path
from bitrix_rag_indexer.state.hashes import sha256_text


def build_payload(
    project: ProjectConfig,
    file_path: Path,
    chunk: Any,
    language: str,
) -> dict[str, Any]:
    """Build the minimal Qdrant payload for a chunk.

    ``rel_path`` is relative to ``project.root``; when ``project.path`` is set
    the prefix is included (e.g. ``bitrix/modules/sale/lib/foo.php``).
    """
    rel_path = compute_rel_path(project, file_path)

    payload: dict[str, Any] = {
        "project": project.project,
        "language": language,
        "rel_path": rel_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "ordinal": chunk.ordinal,
        "content_hash": sha256_text(chunk.text_for_embedding),
        "text": chunk.text,
    }

    chunk_metadata = getattr(chunk, "metadata", {}) or {}
    for key, value in chunk_metadata.items():
        if value in (None, "", [], {}):
            continue
        payload[key] = value

    return payload
