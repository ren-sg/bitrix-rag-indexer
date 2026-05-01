from pathlib import Path

from bitrix_rag_indexer.eval.runner import run_eval


def test_run_eval_builds_group_summary_and_failed_cases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    eval_file = tmp_path / "queries.mvp.local.yaml"
    eval_file.write_text(
        """
queries:
  - id: ajax_hit
    query: "ajax handler"
    group: ajax
    expected:
      path_contains_any:
        - "ajax.php"

  - id: migration_miss
    query: "migration"
    group: migration
    expected:
      path_contains_any:
        - "migration.php"
""",
        encoding="utf-8",
    )

    def fake_search_query(**kwargs):
        query = kwargs["query"]

        if query == "ajax handler":
            return [
                {
                    "payload": {
                        "rel_path": "local/components/example/ajax.php",
                        "text": "BX.ajax handler",
                    },
                    "score": 0.9,
                }
            ]

        return [
            {
                "payload": {
                    "rel_path": "local/components/example/other.php",
                    "text": "not expected",
                },
                "score": 0.5,
            }
        ]

    monkeypatch.setattr(
        "bitrix_rag_indexer.eval.runner.search_query",
        fake_search_query,
    )

    result = run_eval(
        profile="mvp",
        config_dir=tmp_path,
        eval_file=eval_file,
        mode="dense",
    )

    assert result["total"] == 2
    assert result["hit_at_5"] == 1
    assert result["hit_at_10"] == 1

    assert result["by_group"]["ajax"]["total"] == 1
    assert result["by_group"]["ajax"]["hit_at_5"] == 1
    assert result["by_group"]["ajax"]["hit_at_10"] == 1

    assert result["by_group"]["migration"]["total"] == 1
    assert result["by_group"]["migration"]["hit_at_5"] == 0
    assert result["by_group"]["migration"]["hit_at_10"] == 0

    assert len(result["failed_cases"]) == 1
    assert result["failed_cases"][0]["id"] == "migration_miss"
    assert result["failed_cases"][0]["group"] == "migration"


def test_run_eval_uses_ungrouped_when_group_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    eval_file = tmp_path / "queries.mvp.local.yaml"
    eval_file.write_text(
        """
queries:
  - id: no_group
    query: "result modifier"
    expected:
      path_contains_any:
        - "result_modifier.php"
""",
        encoding="utf-8",
    )

    def fake_search_query(**kwargs):
        return [
            {
                "payload": {
                    "rel_path": "local/templates/.default/result_modifier.php",
                    "text": "result modifier",
                },
                "score": 0.9,
            }
        ]

    monkeypatch.setattr(
        "bitrix_rag_indexer.eval.runner.search_query",
        fake_search_query,
    )

    result = run_eval(
        profile="mvp",
        config_dir=tmp_path,
        eval_file=eval_file,
        mode="dense",
    )

    assert result["cases"][0]["group"] == "ungrouped"
    assert result["by_group"]["ungrouped"]["total"] == 1
    assert result["failed_cases"] == []
