# Dense model experiments

## Цель

Эксперименты с dense-моделью нужны, чтобы сравнивать качество поиска без поломки текущего рабочего индекса.

Главное правило:

> Для каждой dense-модели использовать отдельную Qdrant collection.

Это важно, потому что разные embedding-модели могут иметь разную размерность вектора. Например, `BAAI/bge-small-en-v1.5` использует 384-dim vectors, а `intfloat/multilingual-e5-large` — 1024-dim vectors. Если писать разные модели в одну collection, можно получить ошибку размерности или смешать результаты.

## Создать experiment config

Пример для E5:

```bash
uv run python -m bitrix_rag_indexer.experiments.prepare \
  --name multilingual-e5-large \
  --model "intfloat/multilingual-e5-large" \
  --collection "bitrix_code_mvp_multilingual_e5_large" \
  --query-prefix "query: " \
  --document-prefix "passage: " \
  --overwrite
```

После этого появится config dir:

```text
.indexer/experiments/multilingual-e5-large/configs
```

Проверить сгенерированные конфиги:

```bash
cat .indexer/experiments/multilingual-e5-large/configs/embeddings.yaml
cat .indexer/experiments/multilingual-e5-large/configs/qdrant.yaml
cat .indexer/experiments/multilingual-e5-large/configs/chunking.yaml
```

Ожидаемый `embeddings.yaml`:

```yaml
dense:
  provider: fastembed
  model: intfloat/multilingual-e5-large
  batch_size: 32
  cache_enabled: true
  cache_path: .indexer/cache/embeddings.sqlite
  query_prefix: 'query: '
  document_prefix: 'passage: '

sparse:
  enabled: true
  model: Qdrant/bm25
```

Ожидаемый `qdrant.yaml`:

```yaml
url: http://localhost:6333
collection: bitrix_code_mvp_multilingual_e5_large
dense_vector_name: dense
sparse_vector_name: sparse
distance: Cosine
```

Для E5 prefix обязателен:

```text
query_prefix: "query: "
document_prefix: "passage: "
```

Без этого качество retrieval может быть хуже.

## Индексация experiment collection

```bash
uv run bitrix-rag index \
  --profile mvp \
  --source project_local \
  --force \
  --config-dir .indexer/experiments/multilingual-e5-large/configs
```

## Eval experiment collection

```bash
uv run bitrix-rag eval --profile mvp --mode dense --config-dir .indexer/experiments/multilingual-e5-large/configs | grep 'Summary\|Path-only'
uv run bitrix-rag eval --profile mvp --mode qdrant-sparse --config-dir .indexer/experiments/multilingual-e5-large/configs | grep 'Summary\|Path-only'
uv run bitrix-rag eval --profile mvp --mode qdrant-hybrid --config-dir .indexer/experiments/multilingual-e5-large/configs | grep 'Summary\|Path-only'
```

## Проверить, что collection реально существует

```bash
uv run python - <<'PY'
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

for name in [
    "bitrix_code_mvp_sparse",
    "bitrix_code_mvp_multilingual_e5_large",
]:
    try:
        info = client.get_collection(name)
    except Exception as exc:
        print(name)
        print("  NOT FOUND:", exc)
        print()
        continue

    print(name)
    print("  points_count:", info.points_count)
    config = info.config.params
    print("  dense vectors:", getattr(config, "vectors", None))
    print("  sparse vectors:", getattr(config, "sparse_vectors", None))
    print()
PY
```

## Текущий результат E5

```text
intfloat/multilingual-e5-large

Indexing:
dense_embed: 2736.61s
total: 2907.01s
RSS: 3553 MB
chunks: 3659

dense:
Summary: total=40, hit@5=34/40 (85%), hit@10=35/40 (88%)
Path-only Summary: total=40, path_hit@5=35/40 (88%), path_hit@10=36/40 (90%)

qdrant-sparse:
Summary: total=40, hit@5=29/40 (72%), hit@10=32/40 (80%)
Path-only Summary: total=40, path_hit@5=29/40 (72%), path_hit@10=33/40 (82%)

qdrant-hybrid:
Summary: total=40, hit@5=37/40 (92%), hit@10=37/40 (92%)
Path-only Summary: total=40, path_hit@5=37/40 (92%), path_hit@10=37/40 (92%)
```

## Вывод по E5

`intfloat/multilingual-e5-large` — лучший quality-profile для текущего eval.

Плюсы:

- резко лучше dense retrieval;
- лучший qdrant-hybrid результат;
- лучше работает с русскими запросами;
- хорошо подходит для сценария: русский вопрос -> PHP/JS/Bitrix code chunks.

Минусы:

- очень медленная cold indexing;
- высокая память;
- тяжёлая модель;
- не лучший default для частых локальных переиндексаций.

Рекомендация:

```text
default fast profile:
  BAAI/bge-small-en-v1.5 + qdrant-hybrid

quality profile:
  intfloat/multilingual-e5-large + qdrant-hybrid
```

# Replace default model and collection

## Вариант A. Сделать E5 моделью по умолчанию

Изменить основной конфиг:

```text
configs/embeddings.yaml
```

На:

```yaml
dense:
  provider: fastembed
  model: intfloat/multilingual-e5-large
  batch_size: 32
  cache_enabled: true
  cache_path: .indexer/cache/embeddings.sqlite
  query_prefix: 'query: '
  document_prefix: 'passage: '

sparse:
  enabled: true
  model: Qdrant/bm25
```

Изменить основной Qdrant config:

```text
configs/qdrant.yaml
```

На:

```yaml
url: http://localhost:6333
collection: bitrix_code_mvp_multilingual_e5_large
dense_vector_name: dense
sparse_vector_name: sparse
distance: Cosine
```

После этого обычные команды будут использовать E5:

```bash
uv run bitrix-rag search "где вызывается getRows" \
  --source project_local \
  --mode qdrant-hybrid \
  --limit 5 \
  --debug
```

```bash
uv run bitrix-rag eval --profile mvp --mode qdrant-hybrid
```

Если collection `bitrix_code_mvp_multilingual_e5_large` уже проиндексирована, повторный reindex не нужен.

Если менялся chunking или source corpus, переиндексировать:

```bash
uv run bitrix-rag index \
  --profile mvp \
  --source project_local \
  --force
```

## Вариант B. Оставить быстрый default, а E5 держать как quality profile

Основные конфиги оставить на `BAAI/bge-small-en-v1.5` и `bitrix_code_mvp_sparse`.

E5 запускать только явно:

```bash
uv run bitrix-rag search "где создается сделка" \
  --source project_local \
  --mode qdrant-hybrid \
  --limit 10 \
  --config-dir .indexer/experiments/multilingual-e5-large/configs
```

```bash
uv run bitrix-rag eval \
  --profile mvp \
  --mode qdrant-hybrid \
  --config-dir .indexer/experiments/multilingual-e5-large/configs
```

Рекомендованный режим для MVP сейчас:

```text
для быстрых итераций:
  BAAI/bge-small-en-v1.5

для максимального качества:
  intfloat/multilingual-e5-large
```
