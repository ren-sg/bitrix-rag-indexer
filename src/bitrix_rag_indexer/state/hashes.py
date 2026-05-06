import hashlib
import uuid

QDRANT_POINT_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://github.com/ren-sg/bitrix-rag-indexer",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_chunk_id(project: str, rel_path: str, ordinal: int) -> str:
    """
    Qdrant point ID must be UUID or unsigned integer.

    Point ID is stable for the same project + rel_path + ordinal combination.
    Prepending project ensures that identical rel_paths in different projects
    produce different point IDs.
    Content hash is stored separately in payload; it is NOT part of the ID
    so that minor content changes don't invalidate the point slot.
    """
    raw = f"{project}:{rel_path}:{ordinal}"
    return str(uuid.uuid5(QDRANT_POINT_NAMESPACE, raw))

