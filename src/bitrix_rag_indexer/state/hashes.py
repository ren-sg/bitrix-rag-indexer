import hashlib
import uuid


QDRANT_POINT_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://github.com/ren-sg/bitrix-rag-indexer",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_chunk_id(path: str, ordinal: int, text: str | None = None) -> str:
    """
    Qdrant point ID must be UUID or unsigned integer.

    Point ID should be stable for the same source path and chunk ordinal.
    Content hash is stored separately in payload/manifest.
    """
    raw = f"{path}:{ordinal}"
    return str(uuid.uuid5(QDRANT_POINT_NAMESPACE, raw))
