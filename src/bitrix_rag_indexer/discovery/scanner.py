from pathlib import Path

import pathspec


def scan_source(source: dict) -> list[Path]:
    root = Path(source["root"]).expanduser().resolve()
    include = source.get("include", ["**/*"])
    exclude = source.get("exclude", [])

    if not root.exists():
        raise FileNotFoundError(f"Source root does not exist: {root}")

    exclude_spec = pathspec.PathSpec.from_lines("gitwildmatch", exclude)

    files: list[Path] = []

    for pattern in include:
        for path in root.glob(pattern):
            if not path.is_file():
                continue

            rel = path.relative_to(root).as_posix()

            if exclude_spec.match_file(rel):
                continue

            files.append(path)

    return sorted(set(files))
