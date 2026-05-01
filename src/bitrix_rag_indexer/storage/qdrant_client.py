from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, PointIdsList, VectorParams


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

    def search(
        self,
        query_vector: list[float],
        limit: int,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            using=self.vector_name,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                "score": point.score,
                "path": point.payload.get("rel_path") or point.payload.get("path"),
                "text": point.payload.get("text", ""),
                "payload": point.payload,
            }
            for point in response.points
        ]

    def stats(self) -> dict[str, Any]:
        info = self.client.get_collection(self.collection)

        return {
            "collection": self.collection,
            "points_count": info.points_count,
            "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
            "segments_count": getattr(info, "segments_count", None),
            "status": str(info.status),
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

    def delete_points(self, point_ids: list[str]) -> None:
        if not point_ids:
            return

        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(points=point_ids),
            wait=True,
        )
