from pathlib import Path
from fnmatch import fnmatch


def scan_source(source: dict) -> list[Path]:
    root = Path(source["root"]).resolve()
    include = source.get("include", ["**/*"])
    exclude = source.get("exclude", [])

    if not root.exists():
        raise FileNotFoundError(f"Source root does not exist: {root}")

    files: list[Path] = []

    for pattern in include:
        for path in root.glob(pattern):
            if not path.is_file():
                continue

            rel = path.relative_to(root).as_posix()

            if any(fnmatch(rel, ex) for ex in exclude):
                continue

            files.append(path)

    return sorted(set(files))
