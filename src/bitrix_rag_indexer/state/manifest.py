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
        chunk_fts_records: list[dict] | None = None,
    ) -> None:
        path_text = path.as_posix()
        chunk_fts_records = chunk_fts_records or []

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
                delete from chunk_fts
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

            if chunk_fts_records:
                conn.executemany(
                    """
                    insert into chunk_fts (
                        chunk_id,
                        source_name,
                        source_type,
                        language,
                        path,
                        rel_path,
                        text,
                        text_for_embedding
                    )
                    values (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            item["chunk_id"],
                            item["source_name"],
                            item["source_type"],
                            item["language"],
                            item["path"],
                            item["rel_path"],
                            item["text"],
                            item["text_for_embedding"],
                        )
                        for item in chunk_fts_records
                    ],
                )

            conn.commit()

    def list_indexed_paths(self, source_name: str) -> list[Path]:
        with self.state.connect() as conn:
            rows = conn.execute(
                """
                select path
                from indexed_files
                where source_name = ?
                order by path asc
                """,
                (source_name,),
            ).fetchall()

        return [Path(row["path"]) for row in rows]

    def delete_file(self, source_name: str, path: Path) -> None:
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
                delete from indexed_files
                where source_name = ? and path = ?
                """,
                (source_name, path_text),
            )

            if self.sqlite_table_exists(conn, "chunk_fts"):
                conn.execute(
                    """
                    delete from chunk_fts
                    where source_name = ? and path = ?
                    """,
                    (source_name, path_text),
                )

            conn.commit()

    def sqlite_table_exists(self, conn, table_name: str) -> bool:
        row = conn.execute(
            """
            select name
            from sqlite_master
            where type in ('table', 'virtual table')
            and name = ?
            """,
            (table_name,),
        ).fetchone()

        return row is not None
