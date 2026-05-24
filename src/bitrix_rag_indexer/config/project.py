from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bitrix_rag_indexer.config.loader import load_yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Parsed and resolved project configuration."""

    project: str
    """Logical project name used in payload and manifest."""

    root: Path
    """Absolute path to the project root (after env expansion)."""

    scan_root: Path
    """Absolute directory that is actually walked. Equals root / path if path is set."""

    include: list[str]
    """Glob patterns for files to include."""

    exclude: list[str]
    """Merged glob patterns for files to exclude (from exclude + exclude_from)."""

    path: str | None = None
    """Optional sub-directory scanned within root."""

    force_rel_path: bool = False
    """When true, prepend path to rel_path at query time (migration for old payloads)."""


def compute_rel_path(project: ProjectConfig, file_path: Path) -> str:
    """Return rel_path relative to project.root, including path prefix when set."""
    resolved = file_path.resolve()
    if project.path:
        return (Path(project.path) / resolved.relative_to(project.scan_root)).as_posix()
    return resolved.relative_to(project.root).as_posix()


def load_project_config(yaml_path: Path, config_dir: Path | None = None) -> ProjectConfig:
    """Load and validate a single project config file.

    Args:
        yaml_path: Path to the project YAML file.
        config_dir: Base directory used to resolve ``exclude_from`` references.
                    Defaults to ``yaml_path.parent``.
    """
    data = load_yaml(yaml_path)
    base_dir = config_dir if config_dir is not None else yaml_path.parent

    project = data.get("project")
    if not project or not isinstance(project, str):
        raise ValueError(f"Project config '{yaml_path}' is missing required field 'project'")

    raw_root = data.get("root")
    if not raw_root or not isinstance(raw_root, str):
        raise ValueError(f"Project config '{yaml_path}' is missing required field 'root'")

    root = Path(raw_root).expanduser().resolve()

    sub_path = data.get("path", "")
    if sub_path:
        scan_root = (root / sub_path).resolve()
        path: str | None = sub_path
    else:
        scan_root = root
        path = None

    force_rel_path = bool(data.get("force_rel_path", False))

    include: list[str] = data.get("include") or ["**/*"]
    exclude: list[str] = list(data.get("exclude") or [])

    # Merge patterns from external exclude files (e.g. excludes.bitrix.yaml)
    for ref_path_str in (data.get("exclude_from") or []):
        ref_path = (base_dir / ref_path_str).resolve()
        if not ref_path.exists():
            raise FileNotFoundError(
                f"exclude_from file not found: {ref_path} (referenced from {yaml_path})"
            )
        ref_data = load_yaml(ref_path)
        extra = ref_data.get("exclude") or []
        exclude.extend(extra)

    return ProjectConfig(
        project=project,
        root=root,
        scan_root=scan_root,
        include=include,
        exclude=exclude,
        path=path,
        force_rel_path=force_rel_path,
    )


def load_all_projects(config_dir: Path) -> list[ProjectConfig]:
    """Load all project configs from ``config_dir/projects/*.yaml``.

    Files matching ``*.local.yaml`` are also loaded (they are gitignored
    and intended for machine-specific overrides).
    """
    projects_dir = config_dir / "projects"

    if not projects_dir.exists():
        return []

    yaml_files = sorted(projects_dir.glob("*.yaml"))

    projects: list[ProjectConfig] = []
    for yaml_path in yaml_files:
        projects.append(load_project_config(yaml_path, config_dir=config_dir))

    return projects


def get_project(config_dir: Path, project_name: str) -> ProjectConfig:
    """Load a single project by name, searching ``config_dir/projects/``."""
    projects_dir = config_dir / "projects"

    # Try exact filename match first: my_project.yaml / my_project.local.yaml
    for suffix in (f"{project_name}.local.yaml", f"{project_name}.yaml"):
        candidate = projects_dir / suffix
        if candidate.exists():
            cfg = load_project_config(candidate, config_dir=config_dir)
            if cfg.project == project_name:
                return cfg

    # Fall back to scanning all files for a matching project name
    for cfg in load_all_projects(config_dir):
        if cfg.project == project_name:
            return cfg

    raise ValueError(
        f"No project config found for '{project_name}' in {projects_dir}"
    )
