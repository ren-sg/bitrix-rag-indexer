from typing import Iterable

import pytest

from bitrix_rag_indexer.embeddings import dense as dense_module
from bitrix_rag_indexer.embeddings.dense import DenseEmbedder


class FakeVector:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class FakeTextEmbedding:
    instances: list["FakeTextEmbedding"] = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.texts: list[str] = []
        FakeTextEmbedding.instances.append(self)

    def embed(
        self,
        texts: Iterable[str],
        batch_size: int | None = None,
    ) -> Iterable[FakeVector]:
        text_list = list(texts)
        self.texts.extend(text_list)

        for _ in text_list:
            yield FakeVector([1.0, 2.0, 3.0])


def test_dense_embedder_applies_query_and_document_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeTextEmbedding.instances.clear()
    monkeypatch.setattr(dense_module, "TextEmbedding", FakeTextEmbedding)

    embedder = DenseEmbedder(
        {
            "model": "fake-model",
            "batch_size": 32,
            "cache_enabled": False,
            "query_prefix": "query: ",
            "document_prefix": "passage: ",
        }
    )

    model = FakeTextEmbedding.instances[0]

    assert embedder.vector_size == 3
    assert model.texts == ["test"]

    embedder.embed_query("где создается сделка")
    embedder.embed_documents(["class DealRepository {}"])
    embedder.embed(["raw text"])

    assert model.texts == [
        "test",
        "query: где создается сделка",
        "passage: class DealRepository {}",
        "raw text",
    ]
