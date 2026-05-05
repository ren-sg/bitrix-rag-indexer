from pathlib import Path
from typing import Any

from bitrix_rag_indexer.config.loader import load_yaml
from bitrix_rag_indexer.storage.qdrant_client import QdrantStore


def show_stats(config_dir: Path) -> dict[str, Any]:
    qdrant_cfg = load_yaml(config_dir / "qdrant.yaml")
    store = QdrantStore(qdrant_cfg)
    return store.stats()
