from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import yaml


def prepare_dense_experiment_config(
    *,
    name: str,
    dense_model: str,
    collection: str,
    base_config_dir: Path = Path("configs"),
    output_root: Path = Path(".indexer/experiments"),
    overwrite: bool = False,
) -> Path:
    experiment_name = normalize_experiment_name(name)
    target_root = output_root / experiment_name
    target_config_dir = target_root / "configs"

    if not base_config_dir.exists():
        raise FileNotFoundError(f"Base config dir not found: {base_config_dir}")

    if target_config_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Experiment config already exists: {target_config_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(target_config_dir)

    target_config_dir.mkdir(parents=True, exist_ok=True)

    copied_files = copy_yaml_configs(
        source_dir=base_config_dir,
        target_dir=target_config_dir,
    )
    if not copied_files:
        raise ValueError(f"No yaml configs found in: {base_config_dir}")

    update_qdrant_config(
        path=target_config_dir / "qdrant.yaml",
        collection=collection,
    )
    update_embeddings_config(
        path=target_config_dir / "embeddings.yaml",
        dense_model=dense_model,
    )
    write_experiment_readme(
        path=target_root / "README.md",
        name=experiment_name,
        dense_model=dense_model,
        collection=collection,
        config_dir=target_config_dir,
    )

    return target_config_dir


def normalize_experiment_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise ValueError("Experiment name must not be empty")

    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    if any(char not in allowed_chars for char in normalized):
        raise ValueError(
            "Experiment name may contain only latin letters, digits, '-' and '_'"
        )

    return normalized


def copy_yaml_configs(*, source_dir: Path, target_dir: Path) -> list[Path]:
    copied_files: list[Path] = []

    for source_path in sorted(source_dir.glob("*.yaml")):
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
        copied_files.append(target_path)

    return copied_files


def update_qdrant_config(*, path: Path, collection: str) -> None:
    config = read_yaml_mapping(path)
    config["collection"] = collection
    write_yaml_mapping(path, config)


def update_embeddings_config(*, path: Path, dense_model: str) -> None:
    config = read_yaml_mapping(path)

    dense_config = config.get("dense")
    if not isinstance(dense_config, dict):
        dense_config = {}
        config["dense"] = dense_config

    dense_config["model"] = dense_model
    write_yaml_mapping(path, config)


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    return data


def write_yaml_mapping(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            allow_unicode=True,
            sort_keys=False,
        )


def write_experiment_readme(
    *,
    path: Path,
    name: str,
    dense_model: str,
    collection: str,
    config_dir: Path,
) -> None:
    path.write_text(
        "\n".join(
            [
                f"# Dense experiment: {name}",
                "",
                f"Dense model: `{dense_model}`",
                f"Qdrant collection: `{collection}`",
                f"Config dir: `{config_dir}`",
                "",
                "## Commands",
                "",
                "```bash",
                "docker compose up -d qdrant",
                "",
                f"uv run bitrix-rag index --profile mvp --source project_local --force --config-dir {config_dir}",
                "",
                f"uv run bitrix-rag eval --profile mvp --mode dense --config-dir {config_dir}",
                f"uv run bitrix-rag eval --profile mvp --mode qdrant-sparse --config-dir {config_dir}",
                f"uv run bitrix-rag eval --profile mvp --mode qdrant-hybrid --config-dir {config_dir}",
                "```",
                "",
                "## Check Qdrant collection",
                "",
                "```bash",
                "uv run python - <<'PY'",
                "from qdrant_client import QdrantClient",
                "",
                "client = QdrantClient(url='http://localhost:6333')",
                f"name = '{collection}'",
                "",
                "try:",
                "    info = client.get_collection(name)",
                "except Exception as exc:",
                "    print(name)",
                "    print('  NOT FOUND:', exc)",
                "else:",
                "    print(name)",
                "    print('  points_count:', info.points_count)",
                "    config = info.config.params",
                "    print('  dense vectors:', getattr(config, 'vectors', None))",
                "    print('  sparse vectors:', getattr(config, 'sparse_vectors', None))",
                "PY",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an isolated config directory for dense model benchmark."
    )
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--model", required=True, help="Dense embedding model name")
    parser.add_argument("--collection", required=True, help="Qdrant collection name")
    parser.add_argument(
        "--base-config-dir",
        type=Path,
        default=Path("configs"),
        help="Base config directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".indexer/experiments"),
        help="Experiment output root",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing experiment config",
    )

    args = parser.parse_args()

    config_dir = prepare_dense_experiment_config(
        name=args.name,
        dense_model=args.model,
        collection=args.collection,
        base_config_dir=args.base_config_dir,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )

    print(config_dir)


if __name__ == "__main__":
    main()
