from qdrant_client import QdrantClient, models


def ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    existing = get_existing_payload_index_names(client, collection_name)

    keyword_fields = [
        "source_name",
        "source_type",
        "source",
        "area",
        "language",
    ]

    for field_name in keyword_fields:
        if field_name in existing:
            continue

        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )

    if "rel_path" not in existing:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="rel_path",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=64,
                lowercase=True,
            ),
            wait=True,
        )


def get_existing_payload_index_names(
    client: QdrantClient,
    collection_name: str,
) -> set[str]:
    info = client.get_collection(collection_name)
    payload_schema = getattr(info, "payload_schema", None) or {}
    return set(payload_schema.keys())
