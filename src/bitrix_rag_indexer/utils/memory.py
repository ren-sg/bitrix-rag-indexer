import os

import psutil


def get_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def ensure_memory_below_limit(max_memory_mb: int) -> None:
    rss_mb = get_rss_mb()

    if rss_mb > max_memory_mb:
        raise MemoryError(
            f"Memory limit exceeded: rss={rss_mb:.1f} MB, "
            f"limit={max_memory_mb} MB"
        )
