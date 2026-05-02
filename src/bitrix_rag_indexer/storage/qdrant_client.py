from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from bitrix_rag_indexer.storage.payload_indexes import ensure_payload_indexes


class QdrantStore:
    def __init__(self, config: dict, sparse_config: dict | None = None):
        self.url = config["url"]
        self.collection = config["collection"]
        self.vector_name = config.get("dense_vector_name", "dense")
        self.sparse_vector_name = config.get("sparse_vector_name", "sparse")
        self.distance = config.get("distance", "Cosine")

        self.sparse_config = sparse_config or {}
        self.sparse_enabled = bool(self.sparse_config.get("enabled", False))
        self.sparse_model = str(self.sparse_config.get("model", "Qdrant/bm25"))

        self.client = QdrantClient(url=self.url)

    def ensure_collection(self, vector_size: int) -> None:
        collections = self.client.get_collections().collections
        existing = {collection.name for collection in collections}

        if self.collection not in existing:
            create_kwargs: dict[str, Any] = {
                "collection_name": self.collection,
                "vectors_config": {
                    self.vector_name: VectorParams(
                        size=vector_size,
                        distance=self._distance(),
                    )
                },
            }

            if self.sparse_enabled:
                create_kwargs["sparse_vectors_config"] = {
                    self.sparse_vector_name: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                }

            self.client.create_collection(**create_kwargs)

        self.ensure_payload_indexes()

    def upsert(self, points: list[dict[str, Any]]) -> None:
        qdrant_points = []

        for point in points:
            vector: dict[str, Any] = {
                self.vector_name: point["vector"],
            }

            if self.sparse_enabled:
                sparse_text = str(
                    point.get("sparse_text")
                    or point.get("payload", {}).get("text")
                    or ""
                )
                vector[self.sparse_vector_name] = models.Document(
                    text=sparse_text,
                    model=self.sparse_model,
                )

            qdrant_points.append(
                PointStruct(
                    id=point["id"],
                    vector=vector,
                    payload=point["payload"],
                )
            )

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

        return self._format_points(response.points)

    def search_qdrant_hybrid(
        self,
        query_text: str,
        query_vector: list[float],
        limit: int,
        dense_limit: int,
        sparse_limit: int,
        query_filter: models.Filter | None = None,
    ) -> list[dict[str, Any]]:
        if not self.sparse_enabled:
            raise ValueError("Qdrant sparse search is disabled in embeddings.sparse config")

        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using=self.vector_name,
                    limit=dense_limit,
                ),
                models.Prefetch(
                    query=models.Document(
                        text=query_text,
                        model=self.sparse_model,
                    ),
                    using=self.sparse_vector_name,
                    limit=sparse_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results = self._format_points(response.points)
        for item in results:
            item["qdrant_hybrid_score"] = item["score"]
        return results

    def retrieve(self, point_ids: list[str]) -> list[dict[str, Any]]:
        if not point_ids:
            return []

        records = self.client.retrieve(
            collection_name=self.collection,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )

        by_id: dict[str, dict[str, Any]] = {}
        for record in records:
            payload = record.payload or {}
            by_id[str(record.id)] = {
                "id": str(record.id),
                "score": None,
                "path": payload.get("rel_path") or payload.get("path"),
                "text": payload.get("text", ""),
                "payload": payload,
            }

        return [
            by_id[point_id]
            for point_id in point_ids
            if point_id in by_id
        ]

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

    def _format_points(self, points: list[Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for point in points:
            payload = point.payload or {}
            results.append(
                {
                    "id": str(point.id),
                    "score": point.score,
                    "path": payload.get("rel_path") or payload.get("path"),
                    "text": payload.get("text", ""),
                    "payload": payload,
                }
            )

        return results

    def search_sparse(
        self,
        query_text: str,
        limit: int,
        query_filter: models.Filter | None = None,
    ) -> list[dict[str, Any]]:
        if not self.sparse_enabled:
            raise ValueError("Qdrant sparse search is disabled in embeddings.sparse config")

        response = self.client.query_points(
            collection_name=self.collection,
            query=models.Document(
                text=query_text,
                model=self.sparse_model,
            ),
            using=self.sparse_vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results = self._format_points(response.points)
        for item in results:
            item["qdrant_sparse_score"] = item["score"]

        return results
