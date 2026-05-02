from pathlib import Path
from typing import Any
import gc

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from bitrix_rag_indexer.chunking.markdown_chunker import chunk_markdown
from bitrix_rag_indexer.chunking.text_chunker import chunk_text
from bitrix_rag_indexer.chunking.php_chunker import chunk_php
from bitrix_rag_indexer.config.loader import load_yaml
from bitrix_rag_indexer.discovery.scanner import scan_source
from bitrix_rag_indexer.embeddings.dense import DenseEmbedder
from bitrix_rag_indexer.metadata.payload import build_payload
from bitrix_rag_indexer.parsing.detect_language import detect_language
from bitrix_rag_indexer.state.hashes import sha256_text
from bitrix_rag_indexer.state.manifest import Manifest
from bitrix_rag_indexer.storage.qdrant_client import QdrantStore
from bitrix_rag_indexer.utils.batching import batched
from bitrix_rag_indexer.utils.files import file_size, read_text, should_skip_by_size
from bitrix_rag_indexer.utils.memory import ensure_memory_below_limit, get_rss_mb
from bitrix_rag_indexer.utils.profiling import IndexingProfiler, IndexingStats
from bitrix_rag_indexer.search.filters import SearchFilters, build_qdrant_filter
from bitrix_rag_indexer.search.hybrid import rrf_fuse
from bitrix_rag_indexer.search.lexical import LexicalSearchIndex


def index_source(
    profile: str,
    source_name: str | None,
    force: bool,
    dry_run: bool,
    max_files: int | None,
    config_dir: Path,
) -> str:
    profiler = IndexingProfiler()
    stats = IndexingStats()

    sources_cfg = load_yaml(config_dir / f"sources.{profile}.yaml")
    qdrant_cfg = load_yaml(config_dir / "qdrant.yaml")
    embeddings_cfg = load_yaml(config_dir / "embeddings.yaml")
    chunking_cfg = load_yaml(config_dir / "chunking.yaml")
    limits_cfg = load_yaml(config_dir / "limits.yaml")

    limits = limits_cfg["indexing"]
    sources = sources_cfg["sources"]

    if source_name:
        sources = [src for src in sources if src["name"] == source_name]

    if not sources:
        raise ValueError(f"No sources matched: {source_name}")

    embedder = DenseEmbedder(embeddings_cfg["dense"])
    store = QdrantStore(qdrant_cfg, sparse_config=embeddings_cfg.get("sparse"))

    if not dry_run:
        store.ensure_collection(vector_size=embedder.vector_size)

    manifest = Manifest(Path(".indexer/state/index.sqlite"))

    for source in sources:
        with profiler.measure("scan"):
            files = scan_source(source)

        if max_files is not None:
            files = files[:max_files]

        stats.discovered += len(files)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("RSS: {task.fields[rss]} MB"),
        ) as progress:
            task = progress.add_task(
                f"Indexing {source['name']}",
                total=len(files),
                rss=f"{get_rss_mb():.0f}",
            )

            for file_path in files:
                progress.update(
                    task,
                    description=f"{source['name']}: {file_path.name}",
                    rss=f"{get_rss_mb():.0f}",
                )

                stats.scanned += 1

                try:
                    with profiler.measure("memory_guard"):
                        ensure_memory_below_limit(int(limits["max_memory_mb"]))

                    with profiler.measure("file_size"):
                        size = file_size(file_path)
                        too_large = should_skip_by_size(
                            file_path,
                            int(limits["max_file_bytes"]),
                        )

                    stats.bytes += size

                    if too_large:
                        stats.too_large += 1
                        continue

                    if dry_run:
                        continue

                    with profiler.measure("read"):
                        text = read_text(file_path)

                    with profiler.measure("hash"):
                        file_hash = sha256_text(text)

                    with profiler.measure("manifest_check"):
                        unchanged = (
                            not force
                            and manifest.is_file_unchanged(
                                source_name=source["name"],
                                path=file_path,
                                file_hash=file_hash,
                            )
                        )

                    if unchanged:
                        stats.skipped += 1
                        continue

                    with profiler.measure("detect_language"):
                        language = detect_language(file_path)

                    with profiler.measure("chunk"):
                        chunks = make_chunks(
                            text=text,
                            file_path=file_path,
                            language=language,
                            chunking_cfg=chunking_cfg,
                        )

                    if len(chunks) > int(limits["max_chunks_per_file"]):
                        chunks = chunks[: int(limits["max_chunks_per_file"])]

                    with profiler.measure("manifest_read"):
                        old_chunk_ids = manifest.get_chunk_ids(
                            source_name=source["name"],
                            path=file_path,
                        )

                    new_chunk_ids = [chunk.chunk_id for chunk in chunks]

                    if not chunks:
                        with profiler.measure("manifest_replace"):
                            manifest.replace_file(
                                source_name=source["name"],
                                path=file_path,
                                file_hash=file_hash,
                                chunk_ids=[],
                                chunk_fts_records=[],
                            )

                        if old_chunk_ids:
                            with profiler.measure("delete_old_points"):
                                store.delete_points(old_chunk_ids)

                        stats.empty += 1
                        continue

                    index_chunks_in_batches(
                        chunks=chunks,
                        source=source,
                        file_path=file_path,
                        language=language,
                        embedder=embedder,
                        store=store,
                        embed_batch_size=int(limits["embed_batch_size"]),
                        upsert_batch_size=int(limits["upsert_batch_size"]),
                        max_memory_mb=int(limits["max_memory_mb"]),
                        profiler=profiler,
                    )

                    with profiler.measure("fts_records"):
                        chunk_fts_records = build_chunk_fts_records(
                            source=source,
                            file_path=file_path,
                            chunks=chunks,
                            language=language,
                        )

                    with profiler.measure("manifest_replace"):
                        manifest.replace_file(
                            source_name=source["name"],
                            path=file_path,
                            file_hash=file_hash,
                            chunk_ids=new_chunk_ids,
                            chunk_fts_records=chunk_fts_records,
                        )

                    new_chunk_id_set = set(new_chunk_ids)
                    old_chunk_ids_to_delete = [
                        chunk_id
                        for chunk_id in old_chunk_ids
                        if chunk_id not in new_chunk_id_set
                    ]

                    if old_chunk_ids_to_delete:
                        with profiler.measure("delete_old_points"):
                            store.delete_points(old_chunk_ids_to_delete)

                    stats.record_indexed_file(len(chunks))

                    del text
                    del chunks

                    with profiler.measure("gc"):
                        gc.collect()

                except Exception:
                    stats.failed += 1

                    if limits.get("stop_on_error", False):
                        raise

                finally:
                    progress.update(task, rss=f"{get_rss_mb():.0f}")
                    progress.advance(task)

    return format_index_result(stats, profiler)

def build_chunk_fts_records(
    source: dict[str, Any],
    file_path: Path,
    chunks: list[Any],
    language: str,
) -> list[dict[str, Any]]:
    root = Path(source["root"]).resolve()
    rel_path = file_path.resolve().relative_to(root).as_posix()

    return [
        {
            "chunk_id": chunk.chunk_id,
            "source_name": source["name"],
            "source_type": source["type"],
            "language": language,
            "path": file_path.as_posix(),
            "rel_path": rel_path,
            "text": chunk.text,
            "text_for_embedding": chunk.text_for_embedding,
        }
        for chunk in chunks
    ]

def make_chunks(
    text: str,
    file_path: Path,
    language: str,
    chunking_cfg: dict[str, Any],
) -> list[Any]:
    if language == "markdown":
        return chunk_markdown(
            text=text,
            path=file_path,
            config=chunking_cfg["markdown"],
        )

    if language == "php":
        return chunk_php(
            text=text,
            path=file_path,
            language=language,
            config=chunking_cfg.get("php", chunking_cfg["code"]),
        )

    chunk_config = (
        chunking_cfg["code"]
        if language in {"javascript", "typescript", "vue"}
        else chunking_cfg["text"]
    )

    return chunk_text(
        text=text,
        path=file_path,
        language=language,
        config=chunk_config,
    )


def index_chunks_in_batches(
    chunks: list[Any],
    source: dict[str, Any],
    file_path: Path,
    language: str,
    embedder: DenseEmbedder,
    store: QdrantStore,
    embed_batch_size: int,
    upsert_batch_size: int,
    max_memory_mb: int,
    profiler: IndexingProfiler,
) -> None:
    for chunk_batch in batched(chunks, embed_batch_size):
        with profiler.measure("memory_guard"):
            ensure_memory_below_limit(max_memory_mb)

        texts = [chunk.text_for_embedding for chunk in chunk_batch]

        with profiler.measure("dense_embed"):
            vectors = embedder.embed(texts)

        with profiler.measure("memory_guard"):
            ensure_memory_below_limit(max_memory_mb)

        points = []

        with profiler.measure("build_payload"):
            for chunk, vector in zip(chunk_batch, vectors):
                payload = build_payload(
                    source=source,
                    file_path=file_path,
                    chunk=chunk,
                    language=language,
                )

                points.append(
                    {
                        "id": chunk.chunk_id,
                        "vector": vector,
                        "sparse_text": chunk.text_for_embedding,
                        "payload": payload,
                    }
                )

        for point_batch in batched(points, upsert_batch_size):
            with profiler.measure("qdrant_upsert"):
                store.upsert(point_batch)

        del texts
        del vectors
        del points

        with profiler.measure("gc"):
            gc.collect()


def format_index_result(
    stats: IndexingStats,
    profiler: IndexingProfiler,
) -> str:
    return "\n".join(
        [
            stats.format_legacy_summary(),
            "",
            stats.format_details(),
            "",
            profiler.format_timings(),
        ]
    )

def search_query(
    query: str,
    limit: int,
    config_dir: Path,
    score_threshold: float | None = None,
    filters: SearchFilters | None = None,
    mode: str | None = None,
) -> list[dict[str, Any]]:
    qdrant_cfg = load_yaml(config_dir / "qdrant.yaml")
    embeddings_cfg = load_yaml(config_dir / "embeddings.yaml")
    ranking_cfg = load_yaml(config_dir / "ranking.yaml")
    search_cfg = ranking_cfg.get("search", {})
    hybrid_cfg = ranking_cfg.get("hybrid", {})

    default_mode = str(search_cfg.get("default_mode", "dense"))
    dense_candidates = int(hybrid_cfg.get("dense_candidates", 50))
    lexical_candidates = int(hybrid_cfg.get("lexical_candidates", 50))
    rrf_k = int(hybrid_cfg.get("rrf_k", 60))

    store = QdrantStore(qdrant_cfg, sparse_config=embeddings_cfg.get("sparse"))
    store.ensure_payload_indexes()

    mode = (mode or default_mode).lower()

    if mode not in {"dense", "lexical", "hybrid", "qdrant-sparse", "qdrant-hybrid"}:
        raise ValueError(f"Unsupported search mode: {mode}")

    if mode == "lexical":
        return search_lexical_only(
            query=query,
            limit=limit,
            filters=filters,
            store=store,
        )

    embedder = DenseEmbedder(embeddings_cfg["dense"])
    query_vector = embedder.embed([query])[0]
    query_filter = build_qdrant_filter(filters)

    if mode == "qdrant-sparse":
        return store.search_sparse(
            query_text=query,
            limit=limit,
            query_filter=query_filter,
        )

    if mode == "qdrant-hybrid":
        return store.search_qdrant_hybrid(
            query_text=query,
            query_vector=query_vector,
            limit=limit,
            dense_limit=dense_candidates,
            sparse_limit=lexical_candidates,
            query_filter=query_filter,
        )

    dense_results = store.search(
        query_vector=query_vector,
        limit=limit if mode == "dense" else dense_candidates,
        score_threshold=score_threshold,
        query_filter=query_filter,
    )

    if mode == "dense":
        return dense_results

    lexical_results = search_lexical_only(
        query=query,
        limit=lexical_candidates,
        filters=filters,
        store=store,
    )

    return rrf_fuse(
        dense_results=dense_results,
        lexical_results=lexical_results,
        limit=limit,
        k=rrf_k,
    )


def search_lexical_only(
    query: str,
    limit: int,
    filters: SearchFilters | None,
    store: QdrantStore,
) -> list[dict[str, Any]]:
    lexical = LexicalSearchIndex(Path(".indexer/state/index.sqlite"))
    lexical_matches = lexical.search(
        query=query,
        limit=limit,
        filters=filters,
    )

    ids = [item["id"] for item in lexical_matches]
    retrieved = store.retrieve(ids)

    by_id = {
        item["id"]: item
        for item in retrieved
    }

    results: list[dict[str, Any]] = []

    for lexical_item in lexical_matches:
        item_id = lexical_item["id"]

        if item_id not in by_id:
            continue

        result = by_id[item_id]
        result["score"] = lexical_item["lexical_score"]
        result["lexical_score"] = lexical_item["lexical_score"]
        result["lexical_rank"] = lexical_item["rank"]

        results.append(result)

    return results

def show_stats(config_dir: Path) -> dict[str, Any]:
    qdrant_cfg = load_yaml(config_dir / "qdrant.yaml")
    store = QdrantStore(qdrant_cfg)
    return store.stats()

def prune_source(
    profile: str,
    source_name: str,
    config_dir: Path,
    dry_run: bool = False,
) -> str:
    sources_cfg = load_yaml(config_dir / f"sources.{profile}.yaml")
    qdrant_cfg = load_yaml(config_dir / "qdrant.yaml")

    sources = sources_cfg["sources"]
    matched_sources = [
        source
        for source in sources
        if source["name"] == source_name
    ]

    if not matched_sources:
        raise ValueError(f"No source matched: {source_name}")

    source = matched_sources[0]

    manifest = Manifest(Path(".indexer/state/index.sqlite"))
    store = QdrantStore(qdrant_cfg)

    current_paths = {
        path.resolve().as_posix()
        for path in scan_source(source)
    }

    indexed_paths = manifest.list_indexed_paths(source_name=source_name)

    stale_paths = [
        path
        for path in indexed_paths
        if path.resolve().as_posix() not in current_paths
    ]

    deleted_files = 0
    deleted_chunks = 0

    for path in stale_paths:
        chunk_ids = manifest.get_chunk_ids(
            source_name=source_name,
            path=path,
        )

        deleted_files += 1
        deleted_chunks += len(chunk_ids)

        if dry_run:
            continue

        if chunk_ids:
            store.delete_points(chunk_ids)

        manifest.delete_file(
            source_name=source_name,
            path=path,
        )

    mode = "dry-run" if dry_run else "deleted"

    return (
        f"Prune {mode}: "
        f"source={source_name}, "
        f"stale_files={deleted_files}, "
        f"stale_chunks={deleted_chunks}"
    )
