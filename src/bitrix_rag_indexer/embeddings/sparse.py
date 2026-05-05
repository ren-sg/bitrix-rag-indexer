from typing import Any

from fastembed import SparseTextEmbedding


class SparseEmbedder:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.enabled = bool(config.get("enabled", False))
        self.model_name = str(config.get("model", "Qdrant/bm25"))
        self._model: SparseTextEmbedding | None = None

    def _get_model(self) -> SparseTextEmbedding:
        if self._model is None:
            self._model = SparseTextEmbedding(model_name=self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[Any]:
        if not self.enabled or not texts:
            return []

        model = self._get_model()
        # list() forces the generator to evaluate
        return list(model.embed(texts))
