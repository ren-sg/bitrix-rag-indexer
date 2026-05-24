from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from bitrix_rag_indexer.search.result_middleware import parse_use_abs_path


@dataclass(frozen=True)
class McpServerSettings:
    config_dir: Path
    qdrant_url: str | None
    use_abs_path: bool
    default_mode: str
    default_limit: int
    max_limit: int
    max_text_chars: int

    @classmethod
    def from_env(cls) -> McpServerSettings:
        return cls(
            config_dir=Path(os.getenv("BITRIX_RAG_CONFIG_DIR", "configs")),
            qdrant_url=os.getenv("BITRIX_RAG_QDRANT_URL"),
            use_abs_path=parse_use_abs_path(),
            default_mode=os.getenv("BITRIX_RAG_SEARCH_MODE", "qdrant-hybrid"),
            default_limit=int(os.getenv("BITRIX_RAG_DEFAULT_LIMIT", "5")),
            max_limit=int(os.getenv("BITRIX_RAG_MAX_LIMIT", "20")),
            max_text_chars=int(os.getenv("BITRIX_RAG_MAX_TEXT_CHARS", "3000")),
        )
