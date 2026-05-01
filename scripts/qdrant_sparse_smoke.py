from qdrant_client import QdrantClient, models


COLLECTION = "bitrix_rag_sparse_smoke"


def main() -> None:
    client = QdrantClient(url="http://localhost:6333")

    existing = {item.name for item in client.get_collections().collections}
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=4,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )

    docs = [
        {
            "id": 1,
            "text": "BX.ajax отправляет форму компонента через ajax.php",
            "dense": [0.9, 0.1, 0.1, 0.1],
            "path": "local/components/example/form/ajax.php",
        },
        {
            "id": 2,
            "text": "result_modifier.php подготавливает данные шаблона компонента",
            "dense": [0.1, 0.9, 0.1, 0.1],
            "path": "local/components/example/list/result_modifier.php",
        },
        {
            "id": 3,
            "text": "CIBlockElement используется в php классе для работы с инфоблоками",
            "dense": [0.1, 0.1, 0.9, 0.1],
            "path": "local/php_interface/src/Iblock/ElementRepository.php",
        },
    ]

    client.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=doc["id"],
                vector={
                    "dense": doc["dense"],
                    "sparse": models.Document(
                        text=doc["text"],
                        model="Qdrant/bm25",
                    ),
                },
                payload={
                    "text": doc["text"],
                    "rel_path": doc["path"],
                },
            )
            for doc in docs
        ],
    )

    sparse = client.query_points(
        collection_name=COLLECTION,
        query=models.Document(
            text="BX.ajax ajax.php",
            model="Qdrant/bm25",
        ),
        using="sparse",
        limit=3,
        with_payload=True,
        with_vectors=False,
    )

    print("\nSPARSE:")
    for point in sparse.points:
        print(point.id, point.score, point.payload["rel_path"])

    hybrid = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(
                query=[0.9, 0.1, 0.1, 0.1],
                using="dense",
                limit=3,
            ),
            models.Prefetch(
                query=models.Document(
                    text="BX.ajax ajax.php",
                    model="Qdrant/bm25",
                ),
                using="sparse",
                limit=3,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=3,
        with_payload=True,
        with_vectors=False,
    )

    print("\nHYBRID RRF:")
    for point in hybrid.points:
        print(point.id, point.score, point.payload["rel_path"])

    client.delete_collection(COLLECTION)


if __name__ == "__main__":
    main()
