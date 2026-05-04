from __future__ import annotations

from typing import Any

from bitrix_rag_indexer.config.loader import load_yaml
from bitrix_rag_indexer.embeddings.dense import DenseEmbedder
from bitrix_rag_indexer.mcp.settings import McpServerSettings
from bitrix_rag_indexer.search.filters import SearchFilters, build_qdrant_filter
from bitrix_rag_indexer.storage.qdrant_client import QdrantStore


class BitrixCodeSearchService:
    SUPPORTED_MODES = {"dense", "qdrant-sparse", "qdrant-hybrid"}

    def __init__(self, settings: McpServerSettings) -> None:
        self.settings = settings

        self.qdrant_config = load_yaml(settings.config_dir / "qdrant.yaml")
        self.embeddings_config = load_yaml(settings.config_dir / "embeddings.yaml")
        self.ranking_config = load_yaml(settings.config_dir / "ranking.yaml")

        if settings.qdrant_url:
            self.qdrant_config["url"] = settings.qdrant_url

        self.search_config = self.ranking_config.get("search", {})
        self.hybrid_config = self.ranking_config.get("hybrid", {})

        self.default_mode = str(
            self.search_config.get("default_mode", settings.default_mode)
        ).lower()

        self.dense_candidates = int(self.hybrid_config.get("dense_candidates", 50))
        self.sparse_candidates = int(self.hybrid_config.get("lexical_candidates", 50))

        self.embedder = DenseEmbedder(self.embeddings_config["dense"])
        self.store = QdrantStore(
            self.qdrant_config,
            sparse_config=self.embeddings_config.get("sparse"),
        )
        self.store.ensure_payload_indexes()

        self._warmup()

    def search(
        self,
        *,
        query: str,
        limit: int | None = None,
        source: str | None = None,
        lang: str | None = None,
        path: str | None = None,
        mode: str | None = None,
        include_text: bool = True,
        max_text_chars: int | None = None,
    ) -> dict[str, Any]:
        normalized_limit = self._normalize_limit(limit)
        normalized_mode = self._normalize_mode(mode)

        filters = SearchFilters(
            source=source,
            lang=lang,
            path=path,
        )
        query_filter = build_qdrant_filter(filters)

        if normalized_mode == "qdrant-sparse":
            raw_results = self.store.search_sparse(
                query_text=query,
                limit=normalized_limit,
                query_filter=query_filter,
            )
        else:
            query_vector = self.embedder.embed_query(query)

            if normalized_mode == "dense":
                raw_results = self.store.search(
                    query_vector=query_vector,
                    limit=normalized_limit,
                    query_filter=query_filter,
                )
            else:
                raw_results = self.store.search_qdrant_hybrid(
                    query_text=query,
                    query_vector=query_vector,
                    limit=normalized_limit,
                    dense_limit=self.dense_candidates,
                    sparse_limit=self.sparse_candidates,
                    query_filter=query_filter,
                )

        text_limit = max_text_chars or self.settings.max_text_chars

        return {
            "query": query,
            "mode": normalized_mode,
            "limit": normalized_limit,
            "filters": {
                "source": source,
                "lang": filters.lang,
                "path": path,
            },
            "count": len(raw_results),
            "results": [
                self._format_result(
                    item,
                    include_text=include_text,
                    max_text_chars=text_limit,
                )
                for item in raw_results
            ],
        }

    def stats(self) -> dict[str, Any]:
        store_stats = self.store.stats()
        return {
            **store_stats,
            "qdrant_url": self.qdrant_config["url"],
            "collection": self.qdrant_config["collection"],
            "dense_model": self.embeddings_config["dense"]["model"],
            "sparse_enabled": bool(self.embeddings_config.get("sparse", {}).get("enabled")),
            "sparse_model": self.embeddings_config.get("sparse", {}).get("model"),
            "default_mode": self.default_mode,
            "default_limit": self.settings.default_limit,
            "max_limit": self.settings.max_limit,
        }

    def _warmup(self) -> None:
        self.embedder.embed_query("warmup")

    def _normalize_limit(self, limit: int | None) -> int:
        value = limit or self.settings.default_limit
        value = max(1, value)
        return min(value, self.settings.max_limit)

    def _normalize_mode(self, mode: str | None) -> str:
        value = (mode or self.default_mode).lower()

        if value not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported MCP search mode: {value}. "
                f"Supported modes: {sorted(self.SUPPORTED_MODES)}"
            )

        return value

    def _format_result(
        self,
        item: dict[str, Any],
        *,
        include_text: bool,
        max_text_chars: int,
    ) -> dict[str, Any]:
        payload = item.get("payload") or {}
        text = str(item.get("text") or "")

        result = {
            "id": item.get("id"),
            "score": item.get("score"),
            "path": item.get("path"),
            "source_name": payload.get("source_name"),
            "source_type": payload.get("source_type"),
            "language": payload.get("language"),
            "rel_path": payload.get("rel_path"),
            "start_line": payload.get("start_line"),
            "end_line": payload.get("end_line"),
            "php_namespace": payload.get("php_namespace"),
            "php_nearest_type_kind": payload.get("php_nearest_type_kind"),
            "php_nearest_type_name": payload.get("php_nearest_type_name"),
            "php_nearest_function_kind": payload.get("php_nearest_function_kind"),
            "php_nearest_function_name": payload.get("php_nearest_function_name"),
        }

        if include_text:
            result["text"] = self._truncate_text(text, max_text_chars)

        return result

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text

        return text[:max_chars].rstrip() + "\n...<truncated>"
