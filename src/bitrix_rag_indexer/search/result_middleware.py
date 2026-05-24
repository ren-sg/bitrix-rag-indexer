from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from bitrix_rag_indexer.config.project import ProjectConfig, load_all_projects


def parse_use_abs_path(value: str | None = None) -> bool:
    """Parse USE_ABS_PATH env-style boolean."""
    raw = value if value is not None else os.getenv("USE_ABS_PATH", "false")
    return raw.lower() in ("true", "1", "yes")


class ProjectPathRegistry:
    """Index of project configs keyed by logical project name."""

    def __init__(self, config_dir: Path) -> None:
        self._projects = {
            project.project: project
            for project in load_all_projects(config_dir)
        }

    def get(self, project_name: str | None) -> ProjectConfig | None:
        if not project_name:
            return None
        return self._projects.get(project_name)


class SearchResultPathMiddleware:
    """Resolve abs_path and normalize rel_path in search results."""

    def __init__(self, registry: ProjectPathRegistry, use_abs_path: bool) -> None:
        self._registry = registry
        self._use_abs_path = use_abs_path

    @classmethod
    def from_env(cls, config_dir: Path) -> SearchResultPathMiddleware:
        return cls(ProjectPathRegistry(config_dir), parse_use_abs_path())

    def apply(self, item: dict[str, Any]) -> dict[str, Any]:
        payload = item.get("payload")
        if payload is None:
            payload = {}
            has_payload = False
        else:
            has_payload = True

        stored_rel_path = item.get("rel_path") or payload.get("rel_path")
        project_name = item.get("project") or payload.get("project")
        cfg = self._registry.get(project_name)

        if not stored_rel_path or not cfg:
            return item

        if self._use_abs_path:
            if cfg.force_rel_path and cfg.path:
                abs_path = (cfg.root / cfg.path / stored_rel_path).as_posix()
            else:
                abs_path = (cfg.root / stored_rel_path).as_posix()
            item["abs_path"] = abs_path
            if has_payload:
                payload["abs_path"] = abs_path

        if cfg.force_rel_path and cfg.path:
            new_rel_path = (Path(cfg.path) / stored_rel_path).as_posix()
            item["rel_path"] = new_rel_path
            item["path"] = new_rel_path
            if has_payload:
                payload["rel_path"] = new_rel_path
                item["payload"] = payload

        return item
