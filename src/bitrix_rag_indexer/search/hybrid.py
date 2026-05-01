from typing import Any


def rrf_fuse(
    dense_results: list[dict[str, Any]],
    lexical_results: list[dict[str, Any]],
    limit: int,
    k: int = 60,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}

    for rank, item in enumerate(dense_results, start=1):
        item_id = str(item.get("id") or "")

        if not item_id:
            continue

        scores[item_id] = scores.get(item_id, 0.0) + rrf_score(rank, k)
        items[item_id] = {
            **item,
            "dense_rank": rank,
            "dense_score": item.get("score"),
        }

    for rank, item in enumerate(lexical_results, start=1):
        item_id = str(item.get("id") or "")

        if not item_id:
            continue

        scores[item_id] = scores.get(item_id, 0.0) + rrf_score(rank, k)

        if item_id in items:
            items[item_id]["lexical_rank"] = rank
            items[item_id]["lexical_score"] = item.get("lexical_score")
        else:
            items[item_id] = {
                **item,
                "lexical_rank": rank,
                "lexical_score": item.get("lexical_score"),
            }

    ordered_ids = sorted(
        scores.keys(),
        key=lambda item_id: scores[item_id],
        reverse=True,
    )

    results: list[dict[str, Any]] = []

    for item_id in ordered_ids[:limit]:
        item = items[item_id]
        item["score"] = scores[item_id]
        item["hybrid_score"] = scores[item_id]
        results.append(item)

    return results


def rrf_score(rank: int, k: int) -> float:
    return 1.0 / (k + rank)
