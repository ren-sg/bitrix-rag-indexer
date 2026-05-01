import hashlib


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_chunk_id(path: str, ordinal: int, text: str) -> str:
    raw = f"{path}:{ordinal}:{sha256_text(text)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
