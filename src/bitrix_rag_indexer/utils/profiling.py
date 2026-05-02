from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter


PREFERRED_BUCKET_ORDER = [
    "scan",
    "memory_guard",
    "file_size",
    "read",
    "hash",
    "manifest_check",
    "detect_language",
    "chunk",
    "manifest_read",
    "dense_embed",
    "build_payload",
    "qdrant_upsert",
    "fts_records",
    "manifest_replace",
    "delete_old_points",
    "gc",
]


@dataclass
class IndexingStats:
    discovered: int = 0
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    empty: int = 0
    too_large: int = 0
    failed: int = 0
    chunks: int = 0
    bytes: int = 0
    chunk_counts: list[int] = field(default_factory=list)

    def record_indexed_file(self, chunk_count: int) -> None:
        self.indexed += 1
        self.chunks += chunk_count
        self.chunk_counts.append(chunk_count)

    @property
    def scanned_mb(self) -> float:
        return self.bytes / 1024 / 1024

    @property
    def avg_chunks_per_indexed_file(self) -> float:
        if not self.chunk_counts:
            return 0.0
        return self.chunks / len(self.chunk_counts)

    @property
    def max_chunks_per_indexed_file(self) -> int:
        if not self.chunk_counts:
            return 0
        return max(self.chunk_counts)

    def format_legacy_summary(self) -> str:
        return (
            f"Scanned files={self.scanned}, "
            f"indexed={self.indexed}, "
            f"chunks={self.chunks}, "
            f"skipped={self.skipped}, "
            f"empty={self.empty}, "
            f"too_large={self.too_large}, "
            f"failed={self.failed}, "
            f"scanned_mb={self.scanned_mb:.1f}"
        )

    def format_details(self) -> str:
        return "\n".join(
            [
                "Indexing stats:",
                f"  files discovered: {self.discovered}",
                f"  files scanned: {self.scanned}",
                f"  files indexed: {self.indexed}",
                f"  files skipped: {self.skipped}",
                f"  files empty: {self.empty}",
                f"  files too large: {self.too_large}",
                f"  files failed: {self.failed}",
                f"  chunks created: {self.chunks}",
                f"  avg chunks/indexed file: {self.avg_chunks_per_indexed_file:.2f}",
                f"  max chunks/indexed file: {self.max_chunks_per_indexed_file}",
                f"  scanned_mb: {self.scanned_mb:.1f}",
            ]
        )


@dataclass
class IndexingProfiler:
    started_at: float = field(default_factory=perf_counter)
    timings: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, bucket: str) -> Iterator[None]:
        started_at = perf_counter()
        try:
            yield
        finally:
            self.timings[bucket] = self.timings.get(bucket, 0.0) + (
                perf_counter() - started_at
            )

    @property
    def elapsed_seconds(self) -> float:
        return perf_counter() - self.started_at

    def ordered_bucket_names(self) -> list[str]:
        known = [name for name in PREFERRED_BUCKET_ORDER if name in self.timings]
        unknown = sorted(name for name in self.timings if name not in known)
        return known + unknown

    def format_timings(self) -> str:
        lines = ["Timings:"]
        for name in self.ordered_bucket_names():
            lines.append(f"  {name}: {self.timings[name]:.2f}s")
        lines.append(f"  total: {self.elapsed_seconds:.2f}s")
        return "\n".join(lines)
