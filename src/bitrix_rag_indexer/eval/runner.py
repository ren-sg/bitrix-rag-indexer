from pathlib import Path
from typing import Any

from bitrix_rag_indexer.app import search_query
from bitrix_rag_indexer.config.loader import load_yaml
from bitrix_rag_indexer.search.filters import SearchFilters


def run_eval(
    profile: str,
    config_dir: Path,
    eval_file: Path | None = None,
    default_limit: int = 10,
) -> dict[str, Any]:
    eval_path = eval_file or Path("eval") / f"queries.{profile}.yaml"

    data = load_yaml(eval_path)
    cases = data.get("queries", [])

    if not cases:
        return {
            "eval_file": str(eval_path),
            "total": 0,
            "hit_at_5": 0,
            "hit_at_10": 0,
            "hit_at_5_rate": 0.0,
            "hit_at_10_rate": 0.0,
            "cases": [],
        }

    rows: list[dict[str, Any]] = []

    hit_at_5 = 0
    hit_at_10 = 0

    for case in cases:
        case_id = str(case["id"])
        query = str(case["query"])
        limit = int(case.get("limit", default_limit))
        expected_paths = [str(item) for item in case.get("expected_paths", [])]

        filter_data = case.get("filters") or {}

        filters = SearchFilters(
            source=filter_data.get("source"),
            lang=filter_data.get("lang") or filter_data.get("language"),
            path=filter_data.get("path"),
            source_type=filter_data.get("source_type"),
        )

        results = search_query(
            query=query,
            limit=limit,
            config_dir=config_dir,
            score_threshold=case.get("score_threshold"),
            filters=filters,
        )

        first_rank = find_first_expected_rank(
            results=results,
            expected_paths=expected_paths,
        )

        case_hit_at_5 = first_rank is not None and first_rank <= 5
        case_hit_at_10 = first_rank is not None and first_rank <= 10

        if case_hit_at_5:
            hit_at_5 += 1

        if case_hit_at_10:
            hit_at_10 += 1

        rows.append(
            {
                "id": case_id,
                "query": query,
                "expected_paths": expected_paths,
                "first_rank": first_rank,
                "hit_at_5": case_hit_at_5,
                "hit_at_10": case_hit_at_10,
                "top_paths": collect_top_paths(results, limit=min(limit, 5)),
            }
        )

    total = len(cases)

    return {
        "eval_file": str(eval_path),
        "total": total,
        "hit_at_5": hit_at_5,
        "hit_at_10": hit_at_10,
        "hit_at_5_rate": hit_at_5 / total if total else 0.0,
        "hit_at_10_rate": hit_at_10 / total if total else 0.0,
        "cases": rows,
    }


def find_first_expected_rank(
    results: list[dict[str, Any]],
    expected_paths: list[str],
) -> int | None:
    if not expected_paths:
        return None

    for index, item in enumerate(results, start=1):
        rel_path = get_result_path(item)

        if path_matches(rel_path=rel_path, expected_paths=expected_paths):
            return index

    return None


def path_matches(rel_path: str, expected_paths: list[str]) -> bool:
    normalized_path = rel_path.lower()

    return any(
        expected.lower() in normalized_path
        for expected in expected_paths
    )


def collect_top_paths(
    results: list[dict[str, Any]],
    limit: int,
) -> list[str]:
    return [
        get_result_path(item)
        for item in results[:limit]
    ]


def get_result_path(item: dict[str, Any]) -> str:
    payload = item.get("payload") or {}

    return str(
        payload.get("rel_path")
        or item.get("path")
        or ""
    )
