from bitrix_rag_indexer.embeddings.cache import EmbeddingCache, hash_embedding_text


def test_embedding_cache_roundtrip(tmp_path) -> None:
    cache = EmbeddingCache(tmp_path / "embeddings.sqlite")

    text_hash = hash_embedding_text("hello")
    vector = [0.1, 0.2, 0.3]

    assert cache.get_many("test-model", [text_hash]) == {}

    cache.put_many(
        model_name="test-model",
        items=[(text_hash, vector)],
    )

    cached = cache.get_many("test-model", [text_hash])

    assert list(cached) == [text_hash]
    assert cached[text_hash] == vector


def test_embedding_cache_is_model_scoped(tmp_path) -> None:
    cache = EmbeddingCache(tmp_path / "embeddings.sqlite")

    text_hash = hash_embedding_text("same text")

    cache.put_many(
        model_name="model-a",
        items=[(text_hash, [1.0, 2.0])],
    )

    assert cache.get_many("model-b", [text_hash]) == {}
    assert cache.get_many("model-a", [text_hash]) == {
        text_hash: [1.0, 2.0],
    }


def test_embedding_cache_deduplicates_lookup_hashes(tmp_path) -> None:
    cache = EmbeddingCache(tmp_path / "embeddings.sqlite")

    text_hash = hash_embedding_text("duplicate")
    vector = [5.0, 6.0]

    cache.put_many(
        model_name="test-model",
        items=[(text_hash, vector)],
    )

    cached = cache.get_many(
        model_name="test-model",
        text_hashes=[text_hash, text_hash, text_hash],
    )

    assert cached == {text_hash: vector}
