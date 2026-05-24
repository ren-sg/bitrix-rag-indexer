from __future__ import annotations

from pathlib import Path

import pytest

from bitrix_rag_indexer.config.project import ProjectConfig
from bitrix_rag_indexer.search.result_middleware import (
    ProjectPathRegistry,
    SearchResultPathMiddleware,
)


def _bitrix_project(root: Path) -> ProjectConfig:
    return ProjectConfig(
        project="bitrix_modules",
        root=root,
        scan_root=(root / "bitrix" / "modules").resolve(),
        include=["**/*.php"],
        exclude=[],
        path="bitrix/modules",
        force_rel_path=True,
    )


def _local_path_project(root: Path) -> ProjectConfig:
    return ProjectConfig(
        project="example_project",
        root=root,
        scan_root=(root / "local").resolve(),
        include=["**/*.php"],
        exclude=[],
        path="local",
        force_rel_path=False,
    )


class FakeRegistry:
    def __init__(self, projects: dict[str, ProjectConfig]) -> None:
        self._projects = projects

    def get(self, project_name: str | None) -> ProjectConfig | None:
        if not project_name:
            return None
        return self._projects.get(project_name)


def test_force_rel_path_prepends_path(tmp_path: Path) -> None:
    root = tmp_path / "www"
    middleware = SearchResultPathMiddleware(
        FakeRegistry({"bitrix_modules": _bitrix_project(root.resolve())}),
        use_abs_path=False,
    )

    item = {
        "path": "sale/lib/foo.php",
        "payload": {
            "project": "bitrix_modules",
            "rel_path": "sale/lib/foo.php",
        },
    }

    result = middleware.apply(item)

    assert result["rel_path"] == "bitrix/modules/sale/lib/foo.php"
    assert result["path"] == "bitrix/modules/sale/lib/foo.php"
    assert result["payload"]["rel_path"] == "bitrix/modules/sale/lib/foo.php"


def test_abs_path_with_force_rel_path(tmp_path: Path) -> None:
    root = (tmp_path / "www").resolve()
    middleware = SearchResultPathMiddleware(
        FakeRegistry({"bitrix_modules": _bitrix_project(root)}),
        use_abs_path=True,
    )

    item = {
        "payload": {
            "project": "bitrix_modules",
            "rel_path": "sale/lib/foo.php",
        },
    }

    result = middleware.apply(item)

    assert result["abs_path"] == f"{root}/bitrix/modules/sale/lib/foo.php"
    assert result["rel_path"] == "bitrix/modules/sale/lib/foo.php"


def test_abs_path_without_force_rel_path(tmp_path: Path) -> None:
    root = (tmp_path / "proj").resolve()
    middleware = SearchResultPathMiddleware(
        FakeRegistry({"example_project": _local_path_project(root)}),
        use_abs_path=True,
    )

    item = {
        "payload": {
            "project": "example_project",
            "rel_path": "local/foo.php",
        },
    }

    result = middleware.apply(item)

    assert result["abs_path"] == f"{root}/local/foo.php"
    assert result["payload"]["rel_path"] == "local/foo.php"


def test_no_abs_path_when_disabled(tmp_path: Path) -> None:
    root = (tmp_path / "www").resolve()
    middleware = SearchResultPathMiddleware(
        FakeRegistry({"bitrix_modules": _bitrix_project(root)}),
        use_abs_path=False,
    )

    item = {
        "payload": {
            "project": "bitrix_modules",
            "rel_path": "sale/lib/foo.php",
        },
    }

    result = middleware.apply(item)

    assert "abs_path" not in result
    assert result["rel_path"] == "bitrix/modules/sale/lib/foo.php"


def test_unknown_project_passthrough() -> None:
    middleware = SearchResultPathMiddleware(FakeRegistry({}), use_abs_path=True)

    item = {
        "payload": {
            "project": "unknown",
            "rel_path": "sale/lib/foo.php",
        },
    }

    result = middleware.apply(item)

    assert result == item


def test_apply_updates_nested_payload(tmp_path: Path) -> None:
    root = (tmp_path / "www").resolve()
    middleware = SearchResultPathMiddleware(
        FakeRegistry({"bitrix_modules": _bitrix_project(root)}),
        use_abs_path=True,
    )

    item = {
        "path": "sale/lib/foo.php",
        "payload": {
            "project": "bitrix_modules",
            "rel_path": "sale/lib/foo.php",
        },
    }

    result = middleware.apply(item)

    assert result["path"] == "bitrix/modules/sale/lib/foo.php"
    assert result["payload"]["rel_path"] == "bitrix/modules/sale/lib/foo.php"
    assert result["payload"]["abs_path"] == f"{root}/bitrix/modules/sale/lib/foo.php"


def test_search_query_applies_middleware(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bitrix_rag_indexer.search import query as search_query_module

    project_yaml = tmp_path / "projects" / "bitrix_modules.yaml"
    project_yaml.parent.mkdir(parents=True)
    root = (tmp_path / "www").resolve()
    project_yaml.write_text(
        f"project: bitrix_modules\nroot: {root}\npath: bitrix/modules\nforce_rel_path: true\n",
        encoding="utf-8",
    )

    fake_item = {
        "id": "1",
        "score": 0.9,
        "path": "sale/lib/foo.php",
        "text": "code",
        "payload": {
            "project": "bitrix_modules",
            "rel_path": "sale/lib/foo.php",
        },
    }

    class FakeStore:
        def ensure_payload_indexes(self) -> None:
            return None

        def search_sparse(self, **kwargs):
            return [fake_item]

    monkeypatch.setenv("USE_ABS_PATH", "true")
    monkeypatch.setattr(search_query_module, "load_yaml", lambda path: {})
    monkeypatch.setattr(search_query_module, "QdrantStore", lambda *args, **kwargs: FakeStore())

    results = search_query_module.search_query(
        query="foo",
        limit=1,
        config_dir=tmp_path,
        mode="qdrant-sparse",
    )

    assert len(results) == 1
    assert results[0]["rel_path"] == "bitrix/modules/sale/lib/foo.php"
    assert results[0]["abs_path"] == f"{root}/bitrix/modules/sale/lib/foo.php"


def test_project_path_registry_loads_configs(tmp_path: Path) -> None:
    project_yaml = tmp_path / "projects" / "bitrix_modules.yaml"
    project_yaml.parent.mkdir(parents=True)
    root = (tmp_path / "www").resolve()
    project_yaml.write_text(
        f"project: bitrix_modules\nroot: {root}\npath: bitrix/modules\n",
        encoding="utf-8",
    )

    registry = ProjectPathRegistry(tmp_path)
    cfg = registry.get("bitrix_modules")

    assert cfg is not None
    assert cfg.path == "bitrix/modules"
    assert cfg.root == root
