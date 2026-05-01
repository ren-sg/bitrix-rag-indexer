import hashlib
import uuid


QDRANT_POINT_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://github.com/ren-sg/bitrix-rag-indexer",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_chunk_id(path: str, ordinal: int, text: str) -> str:
    """
    Qdrant point ID must be UUID or unsigned integer.

    We still use sha256 internally for stable content hashing,
    but convert final point ID to deterministic UUID.
    """
    content_hash = sha256_text(text)
    raw = f"{path}:{ordinal}:{content_hash}"
    return str(uuid.uuid5(QDRANT_POINT_NAMESPACE, raw))
