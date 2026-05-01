from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, PointIdsList, VectorParams
from qdrant_client import models
from bitrix_rag_indexer.storage.payload_indexes import ensure_payload_indexes


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

        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    self.vector_name: VectorParams(
                        size=vector_size,
                        distance=self._distance(),
                    )
                },
            )

        self.ensure_payload_indexes()

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
        query_filter: models.Filter | None = None,
    ) -> list[dict[str, Any]]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            using=self.vector_name,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )

        results: list[dict[str, Any]] = []

        for point in response.points:
            payload = point.payload or {}

            results.append(
                {
                    "score": point.score,
                    "path": payload.get("rel_path") or payload.get("path"),
                    "text": payload.get("text", ""),
                    "payload": payload,
                }
            )

        return results

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

    def ensure_payload_indexes(self) -> None:
        ensure_payload_indexes(
            client=self.client,
            collection_name=self.collection,
        )
