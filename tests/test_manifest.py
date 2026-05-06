"""Legacy manifest tests updated for project-based schema."""

from pathlib import Path

from bitrix_rag_indexer.state.manifest import Manifest


def make_fts_record(
    *,
    chunk_id: str,
    project: str = "my_project",
    language: str = "php",
    path: str = "/project/local/test.php",
    rel_path: str = "local/test.php",
    text: str = "<?php echo 'test';",
    text_for_embedding: str = "Path: local/test.php\nLanguage: php\n<?php echo 'test';",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "project": project,
        "language": language,
        "path": path,
        "rel_path": rel_path,
        "text": text,
        "text_for_embedding": text_for_embedding,
    }


def fetch_chunk_ids(manifest: Manifest, table: str) -> list[str]:
    with manifest.state.connect() as conn:
        rows = conn.execute(
            f"select chunk_id from {table} order by chunk_id asc"
        ).fetchall()

    return [row["chunk_id"] for row in rows]


def test_replace_file_inserts_file_chunks_and_fts(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "index.sqlite")
    path = Path("/project/local/test.php")

    manifest.replace_file(
        project="my_project",
        path=path,
        file_hash="hash-1",
        chunk_ids=["chunk-1"],
        chunk_fts_records=[
            make_fts_record(chunk_id="chunk-1", path=path.as_posix()),
        ],
    )

    assert manifest.get_chunk_ids("my_project", path) == ["chunk-1"]
    assert fetch_chunk_ids(manifest, "chunk_fts") == ["chunk-1"]


def test_replace_file_replaces_old_file_chunks_and_fts(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "index.sqlite")
    path = Path("/project/local/test.php")

    manifest.replace_file(
        project="my_project",
        path=path,
        file_hash="hash-1",
        chunk_ids=["old-chunk"],
        chunk_fts_records=[
            make_fts_record(chunk_id="old-chunk", path=path.as_posix()),
        ],
    )

    manifest.replace_file(
        project="my_project",
        path=path,
        file_hash="hash-2",
        chunk_ids=["new-chunk-1", "new-chunk-2"],
        chunk_fts_records=[
            make_fts_record(chunk_id="new-chunk-1", path=path.as_posix()),
            make_fts_record(chunk_id="new-chunk-2", path=path.as_posix()),
        ],
    )

    assert manifest.get_chunk_ids("my_project", path) == [
        "new-chunk-1",
        "new-chunk-2",
    ]
    assert fetch_chunk_ids(manifest, "chunk_fts") == [
        "new-chunk-1",
        "new-chunk-2",
    ]


def test_delete_file_removes_manifest_rows_and_fts(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "index.sqlite")
    path = Path("/project/local/test.php")

    manifest.replace_file(
        project="my_project",
        path=path,
        file_hash="hash-1",
        chunk_ids=["chunk-1"],
        chunk_fts_records=[
            make_fts_record(chunk_id="chunk-1", path=path.as_posix()),
        ],
    )

    manifest.delete_file(
        project="my_project",
        path=path,
    )

    assert manifest.get_chunk_ids("my_project", path) == []
    assert fetch_chunk_ids(manifest, "file_chunks") == []
    assert fetch_chunk_ids(manifest, "chunk_fts") == []
