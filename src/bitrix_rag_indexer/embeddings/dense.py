from fastembed import TextEmbedding


class DenseEmbedder:
    def __init__(self, config: dict):
        self.model_name = config.get("model", "BAAI/bge-small-en-v1.5")
        self.batch_size = int(config.get("batch_size", 32))
        self._model = TextEmbedding(model_name=self.model_name)
        self.vector_size = self._detect_vector_size()

    def _detect_vector_size(self) -> int:
        vector = next(self._model.embed(["test"]))
        return len(vector.tolist())

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.embed(texts, batch_size=self.batch_size)
        return [vector.tolist() for vector in vectors]
