from pathlib import Path
from typing import Any
import os

from dotenv import load_dotenv
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    load_dotenv()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")

    return expand_env_vars(data)


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}

    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]

    if isinstance(value, str):
        return os.path.expandvars(value)

    return value
