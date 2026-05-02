from bitrix_rag_indexer.utils.profiling import IndexingProfiler, IndexingStats


def test_indexing_stats_formats_chunk_metrics() -> None:
    stats = IndexingStats()
    stats.discovered = 3
    stats.scanned = 3
    stats.skipped = 1
    stats.bytes = 1024 * 1024

    stats.record_indexed_file(2)
    stats.record_indexed_file(4)

    assert stats.indexed == 2
    assert stats.chunks == 6
    assert stats.avg_chunks_per_indexed_file == 3.0
    assert stats.max_chunks_per_indexed_file == 4

    details = stats.format_details()

    assert "files discovered: 3" in details
    assert "files indexed: 2" in details
    assert "chunks created: 6" in details
    assert "avg chunks/indexed file: 3.00" in details
    assert "max chunks/indexed file: 4" in details


def test_indexing_profiler_accumulates_bucket_timings() -> None:
    profiler = IndexingProfiler()

    with profiler.measure("read"):
        sum(range(10))

    with profiler.measure("read"):
        sum(range(20))

    assert "read" in profiler.timings
    assert profiler.timings["read"] >= 0

    formatted = profiler.format_timings()

    assert "Timings:" in formatted
    assert "read:" in formatted
    assert "total:" in formatted
