from pathlib import Path
import re
import sqlite3
from typing import Any

from bitrix_rag_indexer.search.filters import SearchFilters


TOKEN_RE = re.compile(r"[\wА-Яа-яЁё]+", re.UNICODE)


class LexicalSearchIndex:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def search(
        self,
        query: str,
        limit: int,
        filters: SearchFilters | None = None,
    ) -> list[dict[str, Any]]:
        fts_query = build_fts_query(query)

        if not fts_query:
            return []

        where = ["chunk_fts match ?"]
        params: list[Any] = [fts_query]

        if filters:
            if filters.source:
                where.append("source_name = ?")
                params.append(filters.source)

            if filters.lang:
                where.append("language = ?")
                params.append(filters.lang)

            if filters.source_type:
                where.append("source_type = ?")
                params.append(filters.source_type)

            if filters.path:
                where.append("rel_path like ?")
                params.append(f"%{filters.path}%")

        params.append(limit)

        sql = f"""
            select
                chunk_id,
                source_name,
                source_type,
                language,
                path,
                rel_path,
                bm25(chunk_fts) as rank
            from chunk_fts
            where {" and ".join(where)}
            order by rank asc
            limit ?
        """

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()

        results: list[dict[str, Any]] = []

        for index, row in enumerate(rows, start=1):
            rank = float(row["rank"])

            results.append(
                {
                    "id": row["chunk_id"],
                    "rank": index,
                    "lexical_score": -rank,
                    "path": row["rel_path"],
                }
            )

        return results


def build_fts_query(query: str) -> str:
    tokens = [
        token
        for token in TOKEN_RE.findall(query)
        if len(token) >= 2
    ]

    tokens = dedupe_keep_order(tokens)

    # Ограничиваем, чтобы длинный пользовательский запрос не превращался
    # в огромный FTS expression.
    tokens = tokens[:16]

    if not tokens:
        return ""

    # OR лучше для кода, потому что запросы часто смешивают русский текст,
    # имена классов, пути и JS/PHP идентификаторы.
    return " OR ".join(f'"{escape_fts_token(token)}"' for token in tokens)


def escape_fts_token(token: str) -> str:
    return token.replace('"', '""')


def dedupe_keep_order(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for item in items:
        key = item.casefold()

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result
