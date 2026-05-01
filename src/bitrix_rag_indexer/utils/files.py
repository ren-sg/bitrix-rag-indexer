from pathlib import Path


def file_size(path: Path) -> int:
    return path.stat().st_size


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def should_skip_by_size(path: Path, max_file_bytes: int) -> bool:
    return file_size(path) > max_file_bytes
