"""
Grouped tests for the project-based config refactor.

Groups:
  1. Config loading — required fields, env substitution, path join, exclude_from merge
  2. Payload & identity — payload has project, no absolute path, correct rel_path,
                          IDs differ for same rel_path in different projects
  3. Filters & manifest — build_qdrant_filter uses project, manifest CRUD uses project,
                          lexical search filters by project
"""

import os
from pathlib import Path

import pytest

from bitrix_rag_indexer.config.project import ProjectConfig, compute_rel_path, load_project_config
from bitrix_rag_indexer.metadata.payload import build_payload
from bitrix_rag_indexer.search.filters import SearchFilters, build_qdrant_filter
from bitrix_rag_indexer.state.hashes import stable_chunk_id
from bitrix_rag_indexer.state.manifest import Manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class _FakeChunk:
    """Minimal chunk-like object for build_payload tests."""

    def __init__(self, text: str = "<?php echo 1;", ordinal: int = 1) -> None:
        self.text = text
        self.text_for_embedding = f"Path: test.php\n{text}"
        self.start_line = 1
        self.end_line = 5
        self.ordinal = ordinal
        self.chunk_id = "will-be-set"
        self.metadata = {}


# ---------------------------------------------------------------------------
# 1. Config loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_loads_minimal_project(self, tmp_path: Path) -> None:
        root = tmp_path / "myproject"
        root.mkdir()

        cfg_path = tmp_path / "myproject.yaml"
        _write_yaml(cfg_path, f"project: myproject\nroot: {root}\n")

        cfg = load_project_config(cfg_path)

        assert cfg.project == "myproject"
        assert cfg.root == root.resolve()
        assert cfg.scan_root == root.resolve()

    def test_missing_project_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "myproject"
        root.mkdir()
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(cfg_path, f"root: {root}\n")

        with pytest.raises(ValueError, match="missing required field 'project'"):
            load_project_config(cfg_path)

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(cfg_path, "project: myproject\n")

        with pytest.raises(ValueError, match="missing required field 'root'"):
            load_project_config(cfg_path)

    def test_env_var_substitution_in_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        root = tmp_path / "envproject"
        root.mkdir()
        monkeypatch.setenv("TEST_PROJECT_ROOT", str(root))

        cfg_path = tmp_path / "envproject.yaml"
        _write_yaml(cfg_path, "project: envproject\nroot: ${TEST_PROJECT_ROOT}\n")

        cfg = load_project_config(cfg_path)

        assert cfg.root == root.resolve()

    def test_path_subdir_sets_scan_root(self, tmp_path: Path) -> None:
        root = tmp_path / "www"
        sub = root / "local"
        sub.mkdir(parents=True)

        cfg_path = tmp_path / "proj.yaml"
        _write_yaml(cfg_path, f"project: proj\nroot: {root}\npath: local\n")

        cfg = load_project_config(cfg_path)

        assert cfg.root == root.resolve()
        assert cfg.scan_root == sub.resolve()

    def test_exclude_from_merges_patterns(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()

        shared = tmp_path / "excludes.shared.yaml"
        _write_yaml(shared, "exclude:\n  - '**/*.map'\n  - '**/vendor/**'\n")

        cfg_path = tmp_path / "proj.yaml"
        _write_yaml(
            cfg_path,
            f"project: proj\nroot: {root}\n"
            f"exclude_from:\n  - {shared}\n"
            f"exclude:\n  - '**/cache/**'\n",
        )

        cfg = load_project_config(cfg_path)

        assert "**/*.map" in cfg.exclude
        assert "**/vendor/**" in cfg.exclude
        assert "**/cache/**" in cfg.exclude

    def test_exclude_from_missing_file_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()

        cfg_path = tmp_path / "proj.yaml"
        _write_yaml(
            cfg_path,
            f"project: proj\nroot: {root}\n"
            "exclude_from:\n  - /nonexistent/path.yaml\n",
        )

        with pytest.raises(FileNotFoundError):
            load_project_config(cfg_path)


# ---------------------------------------------------------------------------
# 2. Payload & identity
# ---------------------------------------------------------------------------


class TestPayloadAndIdentity:
    def _make_project(self, tmp_path: Path, name: str) -> ProjectConfig:
        root = tmp_path / name
        root.mkdir(parents=True, exist_ok=True)
        return ProjectConfig(
            project=name,
            root=root.resolve(),
            scan_root=(root / "local").resolve(),
            include=["**/*.php"],
            exclude=[],
            path="local",
        )

    def test_payload_has_project_field(self, tmp_path: Path) -> None:
        project = self._make_project(tmp_path, "alpha")
        file_path = project.scan_root / "Foo.php"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()

        payload = build_payload(project=project, file_path=file_path, chunk=_FakeChunk(), language="php")

        assert payload["project"] == "alpha"

    def test_payload_has_no_absolute_path(self, tmp_path: Path) -> None:
        project = self._make_project(tmp_path, "alpha")
        file_path = project.scan_root / "Foo.php"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()

        payload = build_payload(project=project, file_path=file_path, chunk=_FakeChunk(), language="php")

        # 'path' key must not exist
        assert "path" not in payload
        # source_name, source_type, source must not exist
        for removed in ("source_name", "source_type", "source", "area", "module"):
            assert removed not in payload

    def test_rel_path_is_relative_to_root(self, tmp_path: Path) -> None:
        project = self._make_project(tmp_path, "alpha")
        file_path = project.scan_root / "components" / "Bar.php"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()

        payload = build_payload(project=project, file_path=file_path, chunk=_FakeChunk(), language="php")

        # rel_path is relative to project.root and includes path prefix
        expected = compute_rel_path(project, file_path)
        assert payload["rel_path"] == expected
        assert payload["rel_path"] == "local/components/Bar.php"
        # Must NOT start with '/'
        assert not payload["rel_path"].startswith("/")

    def test_compute_rel_path_includes_path_prefix(self, tmp_path: Path) -> None:
        root = tmp_path / "www"
        scan_root = root / "bitrix" / "modules"
        scan_root.mkdir(parents=True)
        file_path = scan_root / "sale" / "lib" / "Foo.php"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()

        project = ProjectConfig(
            project="bitrix_modules",
            root=root.resolve(),
            scan_root=scan_root.resolve(),
            include=["**/*.php"],
            exclude=[],
            path="bitrix/modules",
        )

        assert compute_rel_path(project, file_path) == "bitrix/modules/sale/lib/Foo.php"

    def test_chunk_ids_differ_for_same_rel_path_in_different_projects(self) -> None:
        rel_path = "local/components/Foo.php"
        ordinal = 1

        id_a = stable_chunk_id(project="project_a", rel_path=rel_path, ordinal=ordinal)
        id_b = stable_chunk_id(project="project_b", rel_path=rel_path, ordinal=ordinal)

        assert id_a != id_b

    def test_chunk_id_stable_for_same_inputs(self) -> None:
        id1 = stable_chunk_id(project="proj", rel_path="local/Foo.php", ordinal=3)
        id2 = stable_chunk_id(project="proj", rel_path="local/Foo.php", ordinal=3)

        assert id1 == id2


# ---------------------------------------------------------------------------
# 3. Filters & manifest
# ---------------------------------------------------------------------------


class TestFiltersAndManifest:
    def test_build_qdrant_filter_uses_project(self) -> None:
        filters = SearchFilters(project="my_project", lang="php")
        qdrant_filter = build_qdrant_filter(filters)

        assert qdrant_filter is not None
        keys = [c.key for c in qdrant_filter.must]
        assert "project" in keys
        assert "language" in keys
        # Old fields must not appear
        assert "source_name" not in keys
        assert "source_type" not in keys

    def test_build_qdrant_filter_empty_returns_none(self) -> None:
        assert build_qdrant_filter(SearchFilters()) is None

    def test_manifest_replace_and_get_uses_project(self, tmp_path: Path) -> None:
        manifest = Manifest(tmp_path / "index.sqlite")
        path = Path("/project/local/test.php")

        manifest.replace_file(
            project="my_project",
            path=path,
            file_hash="hash-1",
            chunk_ids=["chunk-a"],
            chunk_fts_records=[
                {
                    "chunk_id": "chunk-a",
                    "project": "my_project",
                    "language": "php",
                    "path": path.as_posix(),
                    "rel_path": "local/test.php",
                    "text": "<?php",
                    "text_for_embedding": "Path: local/test.php\n<?php",
                }
            ],
        )

        assert manifest.get_chunk_ids("my_project", path) == ["chunk-a"]
        # Different project sees no chunks
        assert manifest.get_chunk_ids("other_project", path) == []

    def test_manifest_project_isolation(self, tmp_path: Path) -> None:
        manifest = Manifest(tmp_path / "index.sqlite")
        path = Path("/shared/local/Foo.php")

        for proj in ("project_a", "project_b"):
            manifest.replace_file(
                project=proj,
                path=path,
                file_hash=f"hash-{proj}",
                chunk_ids=[f"chunk-{proj}"],
            )

        assert manifest.get_chunk_ids("project_a", path) == ["chunk-project_a"]
        assert manifest.get_chunk_ids("project_b", path) == ["chunk-project_b"]

    def test_manifest_is_file_unchanged(self, tmp_path: Path) -> None:
        manifest = Manifest(tmp_path / "index.sqlite")
        path = Path("/project/local/Foo.php")

        manifest.replace_file(
            project="proj",
            path=path,
            file_hash="abc123",
            chunk_ids=["c1"],
        )

        assert manifest.is_file_unchanged(project="proj", path=path, file_hash="abc123")
        assert not manifest.is_file_unchanged(project="proj", path=path, file_hash="different")
        assert not manifest.is_file_unchanged(project="other", path=path, file_hash="abc123")

    def test_manifest_delete_file(self, tmp_path: Path) -> None:
        manifest = Manifest(tmp_path / "index.sqlite")
        path = Path("/project/local/Foo.php")

        manifest.replace_file(project="proj", path=path, file_hash="h1", chunk_ids=["c1"])
        manifest.delete_file(project="proj", path=path)

        assert manifest.get_chunk_ids("proj", path) == []
