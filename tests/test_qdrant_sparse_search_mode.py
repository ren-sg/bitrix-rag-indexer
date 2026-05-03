from pathlib import Path
from typing import Any

import pytest

from bitrix_rag_indexer import app


def test_qdrant_sparse_search_does_not_initialize_dense_embedder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_load_yaml(path: Path) -> dict[str, Any]:
        if path.name == "qdrant.yaml":
            return {
                "url": "http://localhost:6333",
                "collection": "test_collection",
                "dense_vector_name": "dense",
                "sparse_vector_name": "sparse",
                "distance": "Cosine",
            }

        if path.name == "embeddings.yaml":
            return {
                "dense": {
                    "provider": "fastembed",
                    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "batch_size": 32,
                },
                "sparse": {
                    "enabled": True,
                    "model": "Qdrant/bm25",
                },
            }

        if path.name == "ranking.yaml":
            return {
                "search": {
                    "default_mode": "qdrant-hybrid",
                },
                "hybrid": {
                    "dense_candidates": 50,
                    "lexical_candidates": 50,
                    "rrf_k": 60,
                },
            }

        raise AssertionError(f"Unexpected config path: {path}")

    class FailingDenseEmbedder:
        def __init__(self, config: dict[str, Any]) -> None:
            raise AssertionError("DenseEmbedder must not be initialized in qdrant-sparse mode")

    class FakeQdrantStore:
        def __init__(
            self,
            config: dict[str, Any],
            sparse_config: dict[str, Any] | None = None,
        ) -> None:
            self.config = config
            self.sparse_config = sparse_config

        def ensure_payload_indexes(self) -> None:
            return None

        def search_sparse(
            self,
            *,
            query_text: str,
            limit: int,
            query_filter: Any,
        ) -> list[dict[str, Any]]:
            return [
                {
                    "id": "chunk-1",
                    "score": 1.0,
                    "payload": {
                        "path": "local/example.php",
                    },
                }
            ]

    monkeypatch.setattr(app, "load_yaml", fake_load_yaml)
    monkeypatch.setattr(app, "DenseEmbedder", FailingDenseEmbedder)
    monkeypatch.setattr(app, "QdrantStore", FakeQdrantStore)

    results = app.search_query(
        query="BX.ajax",
        limit=10,
        config_dir=tmp_path,
        mode="qdrant-sparse",
    )

    assert results == [
        {
            "id": "chunk-1",
            "score": 1.0,
            "payload": {
                "path": "local/example.php",
            },
        }
    ]
