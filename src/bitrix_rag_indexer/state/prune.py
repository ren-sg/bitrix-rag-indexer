from pathlib import Path

from bitrix_rag_indexer.config.loader import load_yaml
from bitrix_rag_indexer.config.project import get_project
from bitrix_rag_indexer.discovery.scanner import scan_project
from bitrix_rag_indexer.state.manifest import Manifest
from bitrix_rag_indexer.storage.qdrant_client import QdrantStore


def prune_project(
    project_name: str,
    config_dir: Path,
    dry_run: bool = False,
) -> str:
    project = get_project(config_dir, project_name)
    qdrant_cfg = load_yaml(config_dir / "qdrant.yaml")

    manifest = Manifest(Path(".indexer/state/index.sqlite"))
    store = QdrantStore(qdrant_cfg)

    current_paths = {
        path.resolve().as_posix()
        for path in scan_project(project)
    }

    indexed_paths = manifest.list_indexed_paths(project=project_name)

    stale_paths = [
        path
        for path in indexed_paths
        if path.resolve().as_posix() not in current_paths
    ]

    deleted_files = 0
    deleted_chunks = 0

    for path in stale_paths:
        chunk_ids = manifest.get_chunk_ids(
            project=project_name,
            path=path,
        )

        deleted_files += 1
        deleted_chunks += len(chunk_ids)

        if dry_run:
            continue

        if chunk_ids:
            store.delete_points(chunk_ids)

        manifest.delete_file(
            project=project_name,
            path=path,
        )

    mode = "dry-run" if dry_run else "deleted"

    return (
        f"Prune {mode}: "
        f"project={project_name}, "
        f"stale_files={deleted_files}, "
        f"stale_chunks={deleted_chunks}"
    )


# Keep old name as alias for backward compatibility during transition
prune_source = prune_project
