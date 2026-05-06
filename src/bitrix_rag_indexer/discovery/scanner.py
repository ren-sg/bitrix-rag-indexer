from pathlib import Path

import pathspec

from bitrix_rag_indexer.config.project import ProjectConfig


def scan_project(project: ProjectConfig) -> list[Path]:
    """Walk ``project.scan_root`` and return matching, non-excluded files."""
    scan_root = project.scan_root

    if not scan_root.exists():
        raise FileNotFoundError(f"Project scan root does not exist: {scan_root}")

    exclude_spec = pathspec.PathSpec.from_lines("gitwildmatch", project.exclude)

    files: list[Path] = []

    for pattern in project.include:
        for path in scan_root.glob(pattern):
            if not path.is_file():
                continue

            # Exclude matching is done relative to scan_root
            rel = path.relative_to(scan_root).as_posix()

            if exclude_spec.match_file(rel):
                continue

            files.append(path)

    return sorted(set(files))
