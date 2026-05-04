from __future__ import annotations

from pathlib import Path

from fastembed import TextEmbedding

from bitrix_rag_indexer.embeddings.cache import EmbeddingCache, hash_embedding_text
from bitrix_rag_indexer.embeddings.onnx_runtime import OnnxRuntimeEnvironment


class DenseEmbedder:
    def __init__(self, config: dict):
        self.model_name = config.get("model", "BAAI/bge-small-en-v1.5")
        self.batch_size = int(config.get("batch_size", 32))
        self.cache_enabled = bool(config.get("cache_enabled", True))
        self.cache_path = Path(
            config.get("cache_path", ".indexer/cache/embeddings.sqlite")
        )
        self.query_prefix = str(config.get("query_prefix", ""))
        self.document_prefix = str(config.get("document_prefix", ""))
        self.model_cache_dir = normalize_optional_path(config.get("model_cache_dir"))
        self.local_files_only = normalize_optional_bool(config.get("local_files_only"))
        self.cuda = bool(config.get("cuda", False))
        self.providers = normalize_optional_str_list(config.get("providers"))
        self.device_ids = normalize_optional_int_list(config.get("device_ids"))
        self.parallel = normalize_optional_int(config.get("parallel"))
        self.lazy_load = bool(config.get("lazy_load", False))
        self.onnx_log_severity = int(config.get("onnx_log_severity", 3))
        self.preload_cuda_dependencies = bool(
            config.get("preload_cuda_dependencies", True)
        )

        model_kwargs: dict[str, object] = {}

        if self.cuda:
            model_kwargs["cuda"] = True

        if self.providers is not None:
            model_kwargs["providers"] = self.providers

        if self.device_ids is not None:
            model_kwargs["device_ids"] = self.device_ids

        if self.lazy_load:
            model_kwargs["lazy_load"] = True

        if self.model_cache_dir is not None:
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)
            model_kwargs["cache_dir"] = str(self.model_cache_dir)

        if self.local_files_only is not None:
            model_kwargs["local_files_only"] = self.local_files_only

        OnnxRuntimeEnvironment(
            cuda=self.cuda,
            log_severity=self.onnx_log_severity,
            preload_cuda_dependencies=self.preload_cuda_dependencies,
        ).apply()

        self._model = TextEmbedding(
            model_name=self.model_name,
            **model_kwargs,
        )
        self._cache = (
            EmbeddingCache(self.cache_path)
            if self.cache_enabled
            else None
        )
        self.cache_hits = 0
        self.cache_misses = 0
        self.vector_size = self._detect_vector_size()

    def _detect_vector_size(self) -> int:
        vector = next(self._model.embed(["test"]))
        return len(vector.tolist())

    def embed_query(self, text: str) -> list[float]:
        return self.embed([self.query_prefix + text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed([self.document_prefix + text for text in texts])

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self._cache is None:
            return self._embed_uncached(texts)

        text_hashes = [hash_embedding_text(text) for text in texts]
        cached_vectors = self._cache.get_many(
            model_name=self.model_name,
            text_hashes=text_hashes,
        )

        missing_texts: list[str] = []
        missing_hashes: list[str] = []
        seen_missing_hashes: set[str] = set()

        for text, text_hash in zip(texts, text_hashes):
            if text_hash in cached_vectors:
                continue

            if text_hash in seen_missing_hashes:
                continue

            seen_missing_hashes.add(text_hash)
            missing_texts.append(text)
            missing_hashes.append(text_hash)

        new_vectors_by_hash: dict[str, list[float]] = {}

        if missing_texts:
            new_vectors = self._embed_uncached(missing_texts)
            new_vectors_by_hash = dict(zip(missing_hashes, new_vectors))
            self._cache.put_many(
                model_name=self.model_name,
                items=list(new_vectors_by_hash.items()),
            )

        self.cache_hits += len(texts) - len(missing_texts)
        self.cache_misses += len(missing_texts)

        return [
            cached_vectors.get(text_hash) or new_vectors_by_hash[text_hash]
            for text_hash in text_hashes
        ]

    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        embed_kwargs: dict[str, object] = {
            "batch_size": self.batch_size,
        }

        if self.parallel is not None:
            embed_kwargs["parallel"] = self.parallel

        vectors = self._model.embed(texts, **embed_kwargs)
        return [vector.tolist() for vector in vectors]


def normalize_optional_str_list(value: object) -> list[str] | None:
    if value is None:
        return None

    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        result = [str(item) for item in value if str(item)]
        return result or None

    return None


def normalize_optional_int_list(value: object) -> list[int] | None:
    if value is None:
        return None

    if isinstance(value, int):
        return [value]

    if isinstance(value, list):
        result = [int(item) for item in value]
        return result or None

    return None


def normalize_optional_int(value: object) -> int | None:
    if value is None:
        return None

    return int(value)


def normalize_optional_bool(value: object) -> bool | None:
    if value is None:
        return None

    return bool(value)


def normalize_optional_path(value: object) -> Path | None:
    if value is None:
        return None

    path = Path(str(value)).expanduser()
    return path
