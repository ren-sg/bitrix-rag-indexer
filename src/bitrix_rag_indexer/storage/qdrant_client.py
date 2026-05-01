from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantStore:
    def __init__(self, config: dict):
        self.url = config["url"]
        self.collection = config["collection"]
        self.vector_name = config.get("dense_vector_name", "dense")
        self.distance = config.get("distance", "Cosine")
        self.client = QdrantClient(url=self.url)

    def ensure_collection(self, vector_size: int) -> None:
        collections = self.client.get_collections().collections
        existing = {collection.name for collection in collections}

        if self.collection in existing:
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                self.vector_name: VectorParams(
                    size=vector_size,
                    distance=self._distance(),
                )
            },
        )

    def upsert(self, points: list[dict[str, Any]]) -> None:
        qdrant_points = [
            PointStruct(
                id=point["id"],
                vector={self.vector_name: point["vector"]},
                payload=point["payload"],
            )
            for point in points
        ]

        self.client.upsert(
            collection_name=self.collection,
            points=qdrant_points,
        )

    def search(self, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=(self.vector_name, query_vector),
            limit=limit,
            with_payload=True,
        )

        return [
            {
                "score": hit.score,
                "path": hit.payload.get("rel_path") or hit.payload.get("path"),
                "text": hit.payload.get("text", ""),
                "payload": hit.payload,
            }
            for hit in hits
        ]

    def stats(self) -> dict[str, Any]:
        info = self.client.get_collection(self.collection)
        return {
            "collection": self.collection,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status,
        }

    def _distance(self) -> Distance:
        value = self.distance.lower()
        if value == "cosine":
            return Distance.COSINE
        if value == "dot":
            return Distance.DOT
        if value == "euclid":
            return Distance.EUCLID

        raise ValueError(f"Unsupported distance: {self.distance}")
