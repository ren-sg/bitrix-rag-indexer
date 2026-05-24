from pathlib import Path
from typing import Any

from bitrix_rag_indexer.config.loader import load_yaml
from bitrix_rag_indexer.embeddings.dense import DenseEmbedder
from bitrix_rag_indexer.search.filters import SearchFilters, build_qdrant_filter
from bitrix_rag_indexer.search.hybrid import rrf_fuse
from bitrix_rag_indexer.search.result_middleware import SearchResultPathMiddleware
from bitrix_rag_indexer.search.lexical import LexicalSearchIndex
from bitrix_rag_indexer.storage.qdrant_client import QdrantStore


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

    query_filter = build_qdrant_filter(filters)

    if mode == "lexical":
        results = search_lexical_only(
            query=query,
            limit=limit,
            filters=filters,
            store=store,
        )
    elif mode == "qdrant-sparse":
        results = store.search_sparse(
            query_text=query,
            limit=limit,
            query_filter=query_filter,
        )
    else:
        embedder = DenseEmbedder(embeddings_cfg["dense"])
        query_vector = embedder.embed_query(query)

        if mode == "qdrant-hybrid":
            results = store.search_qdrant_hybrid(
                query_text=query,
                query_vector=query_vector,
                limit=limit,
                dense_limit=dense_candidates,
                sparse_limit=lexical_candidates,
                query_filter=query_filter,
            )
        else:
            dense_results = store.search(
                query_vector=query_vector,
                limit=limit if mode == "dense" else dense_candidates,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )

            if mode == "dense":
                results = dense_results
            else:
                lexical_results = search_lexical_only(
                    query=query,
                    limit=lexical_candidates,
                    filters=filters,
                    store=store,
                )
                results = rrf_fuse(
                    dense_results=dense_results,
                    lexical_results=lexical_results,
                    limit=limit,
                    k=rrf_k,
                )

    return _apply_path_middleware(results, config_dir)


def _apply_path_middleware(
    results: list[dict[str, Any]],
    config_dir: Path,
) -> list[dict[str, Any]]:
    if not results:
        return results

    middleware = SearchResultPathMiddleware.from_env(config_dir)
    return [middleware.apply(item) for item in results]


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
