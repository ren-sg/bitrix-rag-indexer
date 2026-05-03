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

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.embed_kwargs: list[dict[str, object]] = []
        FakeTextEmbedding.instances.append(self)

    def embed(self, texts: Iterable[str], **kwargs: object) -> Iterable[FakeVector]:
        self.embed_kwargs.append(kwargs)

        for _ in texts:
            yield FakeVector([1.0, 2.0, 3.0])


def test_dense_embedder_passes_fastembed_gpu_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeTextEmbedding.instances.clear()
    monkeypatch.setattr(dense_module, "TextEmbedding", FakeTextEmbedding)

    embedder = DenseEmbedder(
        {
            "model": "intfloat/multilingual-e5-large",
            "batch_size": 8,
            "cache_enabled": False,
            "query_prefix": "query: ",
            "document_prefix": "passage: ",
            "cuda": True,
            "providers": ["CUDAExecutionProvider"],
            "device_ids": [0],
            "parallel": 1,
            "lazy_load": True,
        }
    )

    model = FakeTextEmbedding.instances[0]

    assert model.kwargs == {
        "model_name": "intfloat/multilingual-e5-large",
        "cuda": True,
        "providers": ["CUDAExecutionProvider"],
        "device_ids": [0],
        "lazy_load": True,
    }

    assert embedder.vector_size == 3

    embedder.embed_documents(["class Example {}"])

    assert model.embed_kwargs[-1] == {
        "batch_size": 8,
        "parallel": 1,
    }
