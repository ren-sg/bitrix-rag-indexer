from pathlib import Path

from bitrix_rag_indexer.config.loader import load_yaml
from bitrix_rag_indexer.discovery.scanner import scan_source
from bitrix_rag_indexer.state.manifest import Manifest
from bitrix_rag_indexer.storage.qdrant_client import QdrantStore


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
