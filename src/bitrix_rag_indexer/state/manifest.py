from pathlib import Path

from bitrix_rag_indexer.state.sqlite import SQLiteState


class Manifest:
    def __init__(self, db_path: Path):
        self.state = SQLiteState(db_path)

    def is_file_unchanged(self, source_name: str, path: Path, file_hash: str) -> bool:
        with self.state.connect() as conn:
            row = conn.execute(
                """
                select file_hash
                from indexed_files
                where source_name = ? and path = ?
                """,
                (source_name, path.as_posix()),
            ).fetchone()

        return row is not None and row["file_hash"] == file_hash

    def get_chunk_ids(self, source_name: str, path: Path) -> list[str]:
        with self.state.connect() as conn:
            rows = conn.execute(
                """
                select chunk_id
                from file_chunks
                where source_name = ? and path = ?
                order by ordinal asc
                """,
                (source_name, path.as_posix()),
            ).fetchall()

        return [row["chunk_id"] for row in rows]

    def replace_file(
        self,
        source_name: str,
        path: Path,
        file_hash: str,
        chunk_ids: list[str],
    ) -> None:
        path_text = path.as_posix()

        with self.state.connect() as conn:
            conn.execute("begin")

            conn.execute(
                """
                delete from file_chunks
                where source_name = ? and path = ?
                """,
                (source_name, path_text),
            )

            conn.execute(
                """
                insert into indexed_files (
                    source_name,
                    path,
                    file_hash,
                    chunk_count,
                    indexed_at
                )
                values (?, ?, ?, ?, current_timestamp)
                on conflict(source_name, path) do update set
                    file_hash = excluded.file_hash,
                    chunk_count = excluded.chunk_count,
                    indexed_at = current_timestamp
                """,
                (source_name, path_text, file_hash, len(chunk_ids)),
            )

            conn.executemany(
                """
                insert into file_chunks (
                    source_name,
                    path,
                    chunk_id,
                    ordinal
                )
                values (?, ?, ?, ?)
                """,
                [
                    (source_name, path_text, chunk_id, ordinal)
                    for ordinal, chunk_id in enumerate(chunk_ids, start=1)
                ],
            )

            conn.commit()
