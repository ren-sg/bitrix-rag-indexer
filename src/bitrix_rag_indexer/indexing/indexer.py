import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from bitrix_rag_indexer.chunking.markdown_chunker import chunk_markdown
from bitrix_rag_indexer.chunking.php import chunk_php
from bitrix_rag_indexer.chunking.text_chunker import chunk_text
from bitrix_rag_indexer.config.loader import load_yaml
from bitrix_rag_indexer.config.project import ProjectConfig, get_project, load_all_projects
from bitrix_rag_indexer.discovery.scanner import scan_project
from bitrix_rag_indexer.embeddings.dense import DenseEmbedder
from bitrix_rag_indexer.embeddings.sparse import SparseEmbedder
from bitrix_rag_indexer.metadata.payload import build_payload
from bitrix_rag_indexer.parsing.detect_language import detect_language
from bitrix_rag_indexer.state.hashes import sha256_text, stable_chunk_id
from bitrix_rag_indexer.state.manifest import Manifest
from bitrix_rag_indexer.storage.qdrant_client import QdrantStore
from bitrix_rag_indexer.utils.batching import batched
from bitrix_rag_indexer.utils.files import file_size, read_text, should_skip_by_size
from bitrix_rag_indexer.utils.memory import ensure_memory_below_limit, get_rss_mb
from bitrix_rag_indexer.utils.profiling import IndexingProfiler, IndexingStats


@dataclass
class PendingIndexJob:
    project: ProjectConfig
    file_path: Path
    language: str
    file_hash: str
    chunks: list[Any]
    old_chunk_ids: list[str]


def pending_chunk_count(jobs: list[PendingIndexJob]) -> int:
    return sum(len(job.chunks) for job in jobs)


class Indexer:
    def __init__(
        self,
        config_dir: Path,
        dry_run: bool,
        force: bool,
    ):
        self.config_dir = config_dir
        self.dry_run = dry_run
        self.force = force

        self.profiler = IndexingProfiler()
        self.stats = IndexingStats()

        qdrant_cfg = load_yaml(config_dir / "qdrant.yaml")
        embeddings_cfg = load_yaml(config_dir / "embeddings.yaml")
        self.chunking_cfg = load_yaml(config_dir / "chunking.yaml")
        limits_cfg = load_yaml(config_dir / "limits.yaml")

        self.limits = limits_cfg["indexing"]
        self.embed_batch_size = int(self.limits["embed_batch_size"])
        self.upsert_batch_size = int(self.limits["upsert_batch_size"])
        self.max_memory_mb = int(self.limits["max_memory_mb"])
        self.flush_chunk_threshold = max(self.embed_batch_size * 8, self.upsert_batch_size)

        self.embedder = DenseEmbedder(embeddings_cfg["dense"])
        self.sparse_embedder = SparseEmbedder(embeddings_cfg.get("sparse", {}))
        self.store = QdrantStore(qdrant_cfg, sparse_config=embeddings_cfg.get("sparse"))

        if not self.dry_run:
            self.store.ensure_collection(vector_size=self.embedder.vector_size)

        self.manifest = Manifest(Path(".indexer/state/index.sqlite"))
        self.pending_jobs: list[PendingIndexJob] = []

    def run(self, project_name: str | None, lang_filter: str | None, max_files: int | None) -> str:
        if project_name:
            projects = [get_project(self.config_dir, project_name)]
        else:
            projects = load_all_projects(self.config_dir)

        if not projects:
            raise ValueError("No project configs found in configs/projects/")

        for project in projects:
            self._process_project(project, lang_filter=lang_filter, max_files=max_files)

        self._flush_pending_jobs()

        return format_index_result(self.stats, self.profiler)

    def _process_project(
        self,
        project: ProjectConfig,
        lang_filter: str | None,
        max_files: int | None,
    ) -> None:
        with self.profiler.measure("scan"):
            files = scan_project(project)

        if max_files is not None:
            files = files[:max_files]

        self.stats.discovered += len(files)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("RSS: {task.fields[rss]} MB"),
        ) as progress:
            task = progress.add_task(
                f"Indexing {project.project}",
                total=len(files),
                rss=f"{get_rss_mb():.0f}",
            )

            for file_path in files:
                progress.update(
                    task,
                    description=f"{project.project}: {file_path.name}",
                    rss=f"{get_rss_mb():.0f}",
                )

                self.stats.scanned += 1

                try:
                    self._process_file(project, file_path, lang_filter=lang_filter)
                except Exception:
                    self.stats.failed += 1

                    if self.limits.get("stop_on_error", False):
                        raise
                finally:
                    progress.update(task, rss=f"{get_rss_mb():.0f}")
                    progress.advance(task)

    def _process_file(
        self,
        project: ProjectConfig,
        file_path: Path,
        lang_filter: str | None,
    ) -> None:
        with self.profiler.measure("memory_guard"):
            ensure_memory_below_limit(self.max_memory_mb)

        with self.profiler.measure("file_size"):
            size = file_size(file_path)
            too_large = should_skip_by_size(
                file_path,
                int(self.limits["max_file_bytes"]),
            )

        self.stats.bytes += size

        if too_large:
            self.stats.too_large += 1
            return

        if self.dry_run:
            return

        with self.profiler.measure("read"):
            text = read_text(file_path)

        with self.profiler.measure("hash"):
            file_hash = sha256_text(text)

        rel_path = file_path.resolve().relative_to(project.root).as_posix()

        with self.profiler.measure("manifest_check"):
            unchanged = (
                not self.force
                and self.manifest.is_file_unchanged(
                    project=project.project,
                    path=file_path,
                    file_hash=file_hash,
                )
            )

        if unchanged:
            self.stats.skipped += 1
            return

        with self.profiler.measure("detect_language"):
            language = detect_language(file_path)

        if lang_filter and language != lang_filter:
            self.stats.skipped += 1
            return

        with self.profiler.measure("chunk"):
            chunks = make_chunks(
                text=text,
                file_path=Path(rel_path),
                language=language,
                chunking_cfg=self.chunking_cfg,
            )

        if len(chunks) > int(self.limits["max_chunks_per_file"]):
            chunks = chunks[: int(self.limits["max_chunks_per_file"])]

        with self.profiler.measure("manifest_read"):
            old_chunk_ids = self.manifest.get_chunk_ids(
                project=project.project,
                path=file_path,
            )

        if not chunks:
            with self.profiler.measure("manifest_replace"):
                self.manifest.replace_file(
                    project=project.project,
                    path=file_path,
                    file_hash=file_hash,
                    chunk_ids=[],
                    chunk_fts_records=[],
                )

            if old_chunk_ids:
                with self.profiler.measure("delete_old_points"):
                    self.store.delete_points(old_chunk_ids)

            self.stats.empty += 1
            return

        self.pending_jobs.append(
            PendingIndexJob(
                project=project,
                file_path=file_path,
                language=language,
                file_hash=file_hash,
                chunks=chunks,
                old_chunk_ids=old_chunk_ids,
            )
        )

        if pending_chunk_count(self.pending_jobs) >= self.flush_chunk_threshold:
            self._flush_pending_jobs()

        del text

        if self.stats.scanned % 100 == 0:
            with self.profiler.measure("gc"):
                gc.collect()

    def _flush_pending_jobs(self) -> None:
        if not self.pending_jobs:
            return

        flattened: list[tuple[PendingIndexJob, Any]] = [
            (job, chunk)
            for job in self.pending_jobs
            for chunk in job.chunks
        ]

        all_points = []

        for chunk_batch in batched(flattened, self.embed_batch_size):
            with self.profiler.measure("memory_guard"):
                ensure_memory_below_limit(self.max_memory_mb)

            texts = [
                chunk.text_for_embedding
                for _, chunk in chunk_batch
            ]

            with self.profiler.measure("dense_embed"):
                vectors = self.embedder.embed_documents(texts)

            with self.profiler.measure("sparse_embed"):
                sparse_vectors = self.sparse_embedder.embed_documents(texts)

            with self.profiler.measure("memory_guard"):
                ensure_memory_below_limit(self.max_memory_mb)

            with self.profiler.measure("build_payload"):
                for i, ((job, chunk), vector) in enumerate(zip(chunk_batch, vectors, strict=True)):
                    rel_path = chunk.chunk_id  # recomputed below via stable_chunk_id
                    rel_path = job.file_path.resolve().relative_to(job.project.root).as_posix()

                    chunk_id = stable_chunk_id(
                        project=job.project.project,
                        rel_path=rel_path,
                        ordinal=chunk.ordinal,
                    )
                    # Keep chunk object's id in sync (used later for manifest)
                    chunk.chunk_id = chunk_id

                    payload = build_payload(
                        project=job.project,
                        file_path=job.file_path,
                        chunk=chunk,
                        language=job.language,
                    )

                    point = {
                        "id": chunk_id,
                        "vector": vector,
                        "sparse_text": chunk.text_for_embedding,
                        "payload": payload,
                    }

                    if sparse_vectors:
                        point["sparse_vector"] = sparse_vectors[i]

                    all_points.append(point)

        for point_batch in batched(all_points, self.upsert_batch_size):
            with self.profiler.measure("qdrant_upsert"):
                self.store.upsert(point_batch)

        for job in self.pending_jobs:
            new_chunk_ids = [chunk.chunk_id for chunk in job.chunks]

            with self.profiler.measure("fts_records"):
                chunk_fts_records = build_chunk_fts_records(
                    project=job.project,
                    file_path=job.file_path,
                    chunks=job.chunks,
                    language=job.language,
                )

            with self.profiler.measure("manifest_replace"):
                self.manifest.replace_file(
                    project=job.project.project,
                    path=job.file_path,
                    file_hash=job.file_hash,
                    chunk_ids=new_chunk_ids,
                    chunk_fts_records=chunk_fts_records,
                )

            new_chunk_id_set = set(new_chunk_ids)
            old_chunk_ids_to_delete = [
                chunk_id
                for chunk_id in job.old_chunk_ids
                if chunk_id not in new_chunk_id_set
            ]

            if old_chunk_ids_to_delete:
                with self.profiler.measure("delete_old_points"):
                    self.store.delete_points(old_chunk_ids_to_delete)

            self.stats.record_indexed_file(len(job.chunks))

        self.pending_jobs.clear()


def index_source(
    project_name: str | None,
    force: bool,
    dry_run: bool,
    max_files: int | None,
    config_dir: Path,
    lang_filter: str | None = None,
) -> str:
    indexer = Indexer(
        config_dir=config_dir,
        dry_run=dry_run,
        force=force,
    )
    return indexer.run(
        project_name=project_name,
        lang_filter=lang_filter,
        max_files=max_files,
    )


def build_chunk_fts_records(
    project: ProjectConfig,
    file_path: Path,
    chunks: list[Any],
    language: str,
) -> list[dict[str, Any]]:
    rel_path = file_path.resolve().relative_to(project.root).as_posix()

    return [
        {
            "chunk_id": chunk.chunk_id,
            "project": project.project,
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
