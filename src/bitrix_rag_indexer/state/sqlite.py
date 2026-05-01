from pathlib import Path
import sqlite3
from collections.abc import Iterator


class SQLiteState:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists indexed_files (
                    source_name text not null,
                    path text not null,
                    file_hash text not null,
                    chunk_count integer not null,
                    indexed_at text not null default current_timestamp,
                    primary key (source_name, path)
                );

                create table if not exists file_chunks (
                    source_name text not null,
                    path text not null,
                    chunk_id text not null,
                    ordinal integer not null,
                    primary key (source_name, path, chunk_id)
                );

                create index if not exists idx_file_chunks_source_path
                    on file_chunks (source_name, path);
                """
            )
