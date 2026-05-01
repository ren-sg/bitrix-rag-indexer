from pathlib import Path
from typing import Any

from bitrix_rag_indexer.chunking.markdown_chunker import chunk_markdown
from bitrix_rag_indexer.config.loader import load_yaml
from bitrix_rag_indexer.discovery.scanner import scan_source
from bitrix_rag_indexer.embeddings.dense import DenseEmbedder
from bitrix_rag_indexer.metadata.payload import build_payload
from bitrix_rag_indexer.storage.qdrant_client import QdrantStore
from bitrix_rag_indexer.utils.files import read_text


def index_source(profile: str, source_name: str | None, config_dir: Path) -> str:
    sources_cfg = load_yaml(config_dir / f"sources.{profile}.yaml")
    qdrant_cfg = load_yaml(config_dir / "qdrant.yaml")
    embeddings_cfg = load_yaml(config_dir / "embeddings.yaml")
    chunking_cfg = load_yaml(config_dir / "chunking.yaml")

    sources = sources_cfg["sources"]
    if source_name:
        sources = [src for src in sources if src["name"] == source_name]

    embedder = DenseEmbedder(embeddings_cfg["dense"])
    store = QdrantStore(qdrant_cfg)
    store.ensure_collection(vector_size=embedder.vector_size)

    total_files = 0
    total_chunks = 0

    for source in sources:
        files = scan_source(source)
        total_files += len(files)

        points = []
        for file_path in files:
            text = read_text(file_path)
            chunks = chunk_markdown(
                text=text,
                path=file_path,
                config=chunking_cfg["markdown"],
            )

            vectors = embedder.embed([chunk.text_for_embedding for chunk in chunks])

            for chunk, vector in zip(chunks, vectors):
                payload = build_payload(source=source, file_path=file_path, chunk=chunk)
                points.append(
                    {
                        "id": chunk.chunk_id,
                        "vector": vector,
                        "payload": payload,
                    }
                )

        if points:
            store.upsert(points)
            total_chunks += len(points)

    return f"Indexed files={total_files}, chunks={total_chunks}"


def search_query(query: str, limit: int, config_dir: Path) -> list[dict[str, Any]]:
    qdrant_cfg = load_yaml(config_dir / "qdrant.yaml")
    embeddings_cfg = load_yaml(config_dir / "embeddings.yaml")

    embedder = DenseEmbedder(embeddings_cfg["dense"])
    store = QdrantStore(qdrant_cfg)

    query_vector = embedder.embed([query])[0]
    return store.search(query_vector=query_vector, limit=limit)


def show_stats(config_dir: Path) -> dict[str, Any]:
    qdrant_cfg = load_yaml(config_dir / "qdrant.yaml")
    store = QdrantStore(qdrant_cfg)
    return store.stats()
