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
from bitrix_rag_indexer.search.filters import SearchFilters, build_qdrant_filter


def index_source(
    profile: str,
    source_name: str | None,
    force: bool,
    dry_run: bool,
    max_files: int | None,
    config_dir: Path,
) -> str:
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
    store = QdrantStore(qdrant_cfg)

    if not dry_run:
        store.ensure_collection(vector_size=embedder.vector_size)

    manifest = Manifest(Path(".indexer/state/index.sqlite"))

    counters = {
        "scanned": 0,
        "indexed": 0,
        "skipped": 0,
        "empty": 0,
        "too_large": 0,
        "failed": 0,
        "chunks": 0,
        "bytes": 0,
    }

    for source in sources:
        files = scan_source(source)

        if max_files is not None:
            files = files[:max_files]

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

                counters["scanned"] += 1

                try:
                    ensure_memory_below_limit(int(limits["max_memory_mb"]))

                    size = file_size(file_path)
                    counters["bytes"] += size

                    if should_skip_by_size(file_path, int(limits["max_file_bytes"])):
                        counters["too_large"] += 1
                        progress.advance(task)
                        continue

                    if dry_run:
                        continue

                    text = read_text(file_path)
                    file_hash = sha256_text(text)

                    if not force and manifest.is_file_unchanged(
                        source_name=source["name"],
                        path=file_path,
                        file_hash=file_hash,
                    ):
                        counters["skipped"] += 1
                        continue

                    language = detect_language(file_path)
                    chunks = make_chunks(
                        text=text,
                        file_path=file_path,
                        language=language,
                        chunking_cfg=chunking_cfg,
                    )

                    if len(chunks) > int(limits["max_chunks_per_file"]):
                        chunks = chunks[: int(limits["max_chunks_per_file"])]

                    old_chunk_ids = manifest.get_chunk_ids(
                        source_name=source["name"],
                        path=file_path,
                    )

                    if old_chunk_ids:
                        store.delete_points(old_chunk_ids)

                    if not chunks:
                        manifest.replace_file(
                            source_name=source["name"],
                            path=file_path,
                            file_hash=file_hash,
                            chunk_ids=[],
                        )
                        counters["empty"] += 1
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
                    )

                    manifest.replace_file(
                        source_name=source["name"],
                        path=file_path,
                        file_hash=file_hash,
                        chunk_ids=[chunk.chunk_id for chunk in chunks],
                    )

                    counters["indexed"] += 1
                    counters["chunks"] += len(chunks)

                    del text
                    del chunks
                    gc.collect()

                except Exception:
                    counters["failed"] += 1

                    if limits.get("stop_on_error", False):
                        raise

                finally:
                    progress.update(task, rss=f"{get_rss_mb():.0f}")
                    progress.advance(task)

    return format_index_result(counters)


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

    chunk_config = (
        chunking_cfg["code"]
        if language in {"php", "javascript", "typescript", "vue"}
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
) -> None:
    for chunk_batch in batched(chunks, embed_batch_size):
        ensure_memory_below_limit(max_memory_mb)
        texts = [chunk.text_for_embedding for chunk in chunk_batch]
        vectors = embedder.embed(texts)

        ensure_memory_below_limit(max_memory_mb)
        points = []

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
                    "payload": payload,
                }
            )

        for point_batch in batched(points, upsert_batch_size):
            store.upsert(point_batch)

        del texts
        del vectors
        del points
        gc.collect()


def format_index_result(counters: dict[str, int]) -> str:
    mb = counters["bytes"] / 1024 / 1024

    return (
        f"Scanned files={counters['scanned']}, "
        f"indexed={counters['indexed']}, "
        f"chunks={counters['chunks']}, "
        f"skipped={counters['skipped']}, "
        f"empty={counters['empty']}, "
        f"too_large={counters['too_large']}, "
        f"failed={counters['failed']}, "
        f"scanned_mb={mb:.1f}"
    )

def search_query(
    query: str,
    limit: int,
    config_dir: Path,
    score_threshold: float | None = None,
    filters: SearchFilters | None = None,
) -> list[dict[str, Any]]:
    qdrant_cfg = load_yaml(config_dir / "qdrant.yaml")
    embeddings_cfg = load_yaml(config_dir / "embeddings.yaml")

    embedder = DenseEmbedder(embeddings_cfg["dense"])
    store = QdrantStore(qdrant_cfg)

    store.ensure_payload_indexes()

    query_vector = embedder.embed([query])[0]
    query_filter = build_qdrant_filter(filters)

    return store.search(
        query_vector=query_vector,
        limit=limit,
        score_threshold=score_threshold,
        query_filter=query_filter,
    )

def show_stats(config_dir: Path) -> dict[str, Any]:
    qdrant_cfg = load_yaml(config_dir / "qdrant.yaml")
    store = QdrantStore(qdrant_cfg)
    return store.stats()
