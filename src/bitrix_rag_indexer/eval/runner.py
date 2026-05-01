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
    mode: str = "dense",
) -> dict[str, Any]:
    eval_path = resolve_eval_path(
        profile=profile,
        eval_file=eval_file,
    )

    data = load_yaml(eval_path)
    cases = data.get("queries", [])

    if not cases:
        return empty_eval_result(eval_path)

    rows: list[dict[str, Any]] = []

    hit_at_5 = 0
    hit_at_10 = 0

    for case in cases:
        case_id = str(case["id"])
        query = str(case["query"])
        limit = int(case.get("limit", default_limit))
        expected = normalize_expected(case)

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
            mode=mode,
        )

        first_match = find_first_expected_match(
            results=results,
            expected=expected,
        )

        first_rank = first_match["rank"] if first_match else None
        matched_path = first_match["path"] if first_match else None

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
                "first_rank": first_rank,
                "matched_path": matched_path,
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


def empty_eval_result(eval_path: Path) -> dict[str, Any]:
    return {
        "eval_file": str(eval_path),
        "total": 0,
        "hit_at_5": 0,
        "hit_at_10": 0,
        "hit_at_5_rate": 0.0,
        "hit_at_10_rate": 0.0,
        "cases": [],
    }


def normalize_expected(case: dict[str, Any]) -> dict[str, list[str]]:
    expected_raw = case.get("expected") or {}

    expected = {
        "path_contains_any": to_string_list(
            expected_raw.get("path_contains_any")
        ),
        "path_contains_all": to_string_list(
            expected_raw.get("path_contains_all")
        ),
        "path_not_contains": to_string_list(
            expected_raw.get("path_not_contains")
        ),
        "text_contains_any": to_string_list(
            expected_raw.get("text_contains_any")
        ),
        "text_contains_all": to_string_list(
            expected_raw.get("text_contains_all")
        ),
        "text_not_contains": to_string_list(
            expected_raw.get("text_not_contains")
        ),
    }

    # Backward compatibility with the first MVP eval format.
    legacy_expected_paths = to_string_list(case.get("expected_paths"))

    if legacy_expected_paths and not expected["path_contains_any"]:
        expected["path_contains_any"] = legacy_expected_paths

    return expected


def to_string_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    return [str(item) for item in value]


def find_first_expected_match(
    results: list[dict[str, Any]],
    expected: dict[str, list[str]],
) -> dict[str, Any] | None:
    if expected_is_empty(expected):
        return None

    for index, item in enumerate(results, start=1):
        if result_matches_expected(item=item, expected=expected):
            return {
                "rank": index,
                "path": get_result_path(item),
                "score": item.get("score"),
            }

    return None


def expected_is_empty(expected: dict[str, list[str]]) -> bool:
    return not any(expected.values())


def result_matches_expected(
    item: dict[str, Any],
    expected: dict[str, list[str]],
) -> bool:
    path = normalize_text(get_result_path(item))
    text = normalize_text(get_result_text(item))

    if not contains_any(path, expected["path_contains_any"]):
        return False

    if not contains_all(path, expected["path_contains_all"]):
        return False

    if contains_any(path, expected["path_not_contains"], default=False):
        return False

    if not contains_any(text, expected["text_contains_any"]):
        return False

    if not contains_all(text, expected["text_contains_all"]):
        return False

    if contains_any(text, expected["text_not_contains"], default=False):
        return False

    return True


def contains_any(
    haystack: str,
    needles: list[str],
    default: bool = True,
) -> bool:
    if not needles:
        return default

    return any(
        normalize_text(needle) in haystack
        for needle in needles
    )


def contains_all(
    haystack: str,
    needles: list[str],
) -> bool:
    if not needles:
        return True

    return all(
        normalize_text(needle) in haystack
        for needle in needles
    )


def normalize_text(value: str) -> str:
    return value.casefold()


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


def get_result_text(item: dict[str, Any]) -> str:
    payload = item.get("payload") or {}

    return str(
        item.get("text")
        or payload.get("text")
        or ""
    )

def resolve_eval_path(
    profile: str,
    eval_file: Path | None,
) -> Path:
    if eval_file is not None:
        return eval_file

    local_path = Path("eval") / f"queries.{profile}.local.yaml"

    if local_path.exists():
        return local_path

    default_path = Path("eval") / f"queries.{profile}.yaml"

    if default_path.exists():
        return default_path

    return Path("eval") / f"queries.{profile}.example.yaml"
