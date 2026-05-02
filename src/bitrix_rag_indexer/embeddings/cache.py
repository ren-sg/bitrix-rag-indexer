from __future__ import annotations

from array import array
from pathlib import Path
import hashlib
import sqlite3


def hash_embedding_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def get_many(
        self,
        model_name: str,
        text_hashes: list[str],
    ) -> dict[str, list[float]]:
        if not text_hashes:
            return {}

        result: dict[str, list[float]] = {}

        with self._connect() as conn:
            for hash_batch in self._batched_unique(text_hashes, batch_size=500):
                placeholders = ",".join("?" for _ in hash_batch)
                rows = conn.execute(
                    f"""
                    select text_hash, dim, vector
                    from dense_embedding_cache
                    where model_name = ?
                      and text_hash in ({placeholders})
                    """,
                    [model_name, *hash_batch],
                ).fetchall()

                for row in rows:
                    result[row["text_hash"]] = self._decode_vector(
                        blob=row["vector"],
                        dim=int(row["dim"]),
                    )

        return result

    def put_many(
        self,
        model_name: str,
        items: list[tuple[str, list[float]]],
    ) -> None:
        if not items:
            return

        with self._connect() as conn:
            conn.executemany(
                """
                insert into dense_embedding_cache (
                    model_name,
                    text_hash,
                    dim,
                    vector,
                    created_at
                )
                values (?, ?, ?, ?, current_timestamp)
                on conflict(model_name, text_hash) do update set
                    dim = excluded.dim,
                    vector = excluded.vector,
                    created_at = current_timestamp
                """,
                [
                    (
                        model_name,
                        text_hash,
                        len(vector),
                        self._encode_vector(vector),
                    )
                    for text_hash, vector in items
                ],
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists dense_embedding_cache (
                    model_name text not null,
                    text_hash text not null,
                    dim integer not null,
                    vector blob not null,
                    created_at text not null default current_timestamp,
                    primary key (model_name, text_hash)
                )
                """
            )

    def _encode_vector(self, vector: list[float]) -> bytes:
        return array("f", vector).tobytes()

    def _decode_vector(self, blob: bytes, dim: int) -> list[float]:
        values = array("f")
        values.frombytes(blob)

        if len(values) != dim:
            raise ValueError(
                f"Cached embedding vector has invalid dim: expected={dim}, actual={len(values)}"
            )

        return values.tolist()

    def _batched_unique(
        self,
        values: list[str],
        batch_size: int,
    ) -> list[list[str]]:
        unique_values = list(dict.fromkeys(values))
        return [
            unique_values[index : index + batch_size]
            for index in range(0, len(unique_values), batch_size)
        ]
