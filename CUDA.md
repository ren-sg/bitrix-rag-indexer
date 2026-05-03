# NVIDIA CUDA profile for FastEmbed / E5

## Цель

Добавить отдельный CUDA-профиль для индексации dense embeddings на NVIDIA GPU, не ломая обычный CPU-профиль и уже созданные Qdrant collections.

Текущая рабочая схема:

```text
configs/
  основной CPU/default profile

.indexer/experiments/multilingual-e5-large/configs/
  CPU E5 quality profile

.indexer/experiments/multilingual-e5-large-cuda3060/configs/
  NVIDIA CUDA E5 experiment/profile
```

Главное правило:

> Не смешивать CPU и CUDA окружения в одной `.venv`.

Для CUDA использовать отдельную venv:

```text
.venv       обычный CPU режим
.venv-cuda  NVIDIA CUDA режим
```

---

## Почему отдельная venv

В обычном CPU-окружении нужны:

```text
fastembed
onnxruntime
```

В NVIDIA CUDA-окружении нужны:

```text
fastembed-gpu
onnxruntime-gpu
```

CPU и GPU ONNX-пакеты могут конфликтовать, потому что ставят файлы в один и тот же Python package namespace `onnxruntime`.

Если в окружении одновременно окажутся CPU/GPU варианты, `CUDAExecutionProvider` может исчезнуть, и FastEmbed будет работать только на CPU.

---

## AMD / Radeon

Для AMD GPU ничего не трогаем в MVP.

На AMD desktop использовать обычное CPU-окружение:

```bash
rm -rf .venv
uv venv .venv --python 3.12
uv sync
uv run pytest -q
```

Проверка CPU providers:

```bash
uv run python - <<'PY'
import onnxruntime as ort
from fastembed import TextEmbedding

print("ORT providers:", ort.get_available_providers())
print("FastEmbed import: OK")
PY
```

Нормально для CPU:

```text
ORT providers: ['AzureExecutionProvider', 'CPUExecutionProvider']
FastEmbed import: OK
```

На AMD не ставить в основную `.venv`:

```bash
fastembed-gpu
onnxruntime-gpu
```

Для AMD GPU нужен отдельный backend позже, например:

```text
sentence-transformers + PyTorch ROCm
или отдельный ONNX Runtime / MIGraphX backend
```

Это не часть текущего MVP.

---

# NVIDIA CUDA setup

## 1. Проверить NVIDIA

```bash
nvidia-smi
```

Если команда не работает, CUDA-профиль не настраивать.

---

## 2. Создать отдельную CUDA venv

Из корня проекта:

```bash
deactivate 2>/dev/null || true

uv python install 3.12
uv venv .venv-cuda --python 3.12

source .venv-cuda/bin/activate
```

Проверить Python:

```bash
python -V
```

Ожидаемо:

```text
Python 3.12.x
```

---

## 3. Установить проект и CUDA FastEmbed

В активированной `.venv-cuda`:

```bash
uv pip install -e .
uv pip uninstall fastembed onnxruntime -y
uv pip install fastembed-gpu
```

Проверить установленные пакеты:

```bash
python - <<'PY'
import importlib.metadata as md

for pkg in ["fastembed", "fastembed-gpu", "onnxruntime", "onnxruntime-gpu"]:
    try:
        print(pkg, md.version(pkg))
    except md.PackageNotFoundError:
        print(pkg, "not installed")
PY
```

Для CUDA-окружения желательно:

```text
fastembed-gpu installed
onnxruntime-gpu installed
onnxruntime not installed
```

---

## 4. Проверить ONNX Runtime providers

```bash
python - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY
```

Нужно увидеть:

```text
CUDAExecutionProvider
```

Пример нормального результата:

```text
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Если видишь только:

```text
['AzureExecutionProvider', 'CPUExecutionProvider']
```

значит CUDA backend не подхватился. Индексацию CUDA collection не запускать.

---

## 5. Проверить FastEmbed на CUDA

```bash
python - <<'PY'
from fastembed import TextEmbedding

model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    providers=["CUDAExecutionProvider"],
)

print(model.model.model.get_providers())
PY
```

Ожидаемо:

```text
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Если получаешь ошибку:

```text
Provider CUDAExecutionProvider is not available
```

значит окружение всё ещё CPU-only.

---

# Создание отдельного CUDA experiment config

## 1. Создать experiment config

В активированной `.venv-cuda`:

```bash
python -m bitrix_rag_indexer.experiments.prepare \
  --name multilingual-e5-large-cuda3060 \
  --model "intfloat/multilingual-e5-large" \
  --collection "bitrix_code_mvp_multilingual_e5_large_cuda3060" \
  --query-prefix "query: " \
  --document-prefix "passage: " \
  --cache-path ".indexer/cache/embeddings_cuda3060.sqlite" \
  --cuda \
  --provider "CUDAExecutionProvider" \
  --device-id 0 \
  --overwrite
```

Создастся:

```text
.indexer/experiments/multilingual-e5-large-cuda3060/configs
```

---

## 2. Проверить generated config

```bash
cat .indexer/experiments/multilingual-e5-large-cuda3060/configs/embeddings.yaml
cat .indexer/experiments/multilingual-e5-large-cuda3060/configs/qdrant.yaml
```

Ожидаемый `embeddings.yaml`:

```yaml
dense:
  provider: fastembed
  model: intfloat/multilingual-e5-large
  batch_size: 32
  cache_enabled: true
  cache_path: .indexer/cache/embeddings_cuda3060.sqlite
  query_prefix: 'query: '
  document_prefix: 'passage: '
  cuda: true
  providers:
    - CUDAExecutionProvider
  device_ids:
    - 0

sparse:
  enabled: true
  model: Qdrant/bm25
```

Ожидаемый `qdrant.yaml`:

```yaml
url: http://localhost:6333
collection: bitrix_code_mvp_multilingual_e5_large_cuda3060
dense_vector_name: dense
sparse_vector_name: sparse
distance: Cosine
```

Важно:

```text
cache_path отдельный:
.indexer/cache/embeddings_cuda3060.sqlite

collection отдельная:
bitrix_code_mvp_multilingual_e5_large_cuda3060
```

Это нужно, чтобы не смешивать CPU/GPU benchmark и не портить уже готовые collections.

---

# Индексация CUDA collection

## 1. Запустить Qdrant

```bash
docker compose up -d qdrant
```

## 2. Индексировать

```bash
python -m bitrix_rag_indexer.cli index \
  --profile mvp \
  --source project_local \
  --force \
  --config-dir .indexer/experiments/multilingual-e5-large-cuda3060/configs
```

Если entrypoint доступен в CUDA venv, можно так:

```bash
bitrix-rag index \
  --profile mvp \
  --source project_local \
  --force \
  --config-dir .indexer/experiments/multilingual-e5-large-cuda3060/configs
```

Сравнивать с CPU E5 baseline:

```text
CPU E5:
dense_embed: 2736.61s
total: 2907.01s
RSS: 3553 MB
```

CUDA E5 должен дать меньшее `dense_embed`, если CUDA provider реально используется.

---

## Если на RTX 3060 6GB будет OOM

Открой:

```text
.indexer/experiments/multilingual-e5-large-cuda3060/configs/embeddings.yaml
```

Снизь:

```yaml
batch_size: 32
```

до:

```yaml
batch_size: 8
```

Если снова OOM:

```yaml
batch_size: 4
```

Потом повторить index.

---

# Eval CUDA collection

```bash
bitrix-rag eval --profile mvp --mode dense --config-dir .indexer/experiments/multilingual-e5-large-cuda3060/configs | grep 'Summary\|Path-only'
bitrix-rag eval --profile mvp --mode qdrant-sparse --config-dir .indexer/experiments/multilingual-e5-large-cuda3060/configs | grep 'Summary\|Path-only'
bitrix-rag eval --profile mvp --mode qdrant-hybrid --config-dir .indexer/experiments/multilingual-e5-large-cuda3060/configs | grep 'Summary\|Path-only'
```

Качество должно быть примерно как у CPU E5:

```text
dense:
hit@10 ~= 35/40

qdrant-hybrid:
hit@10 ~= 37/40
path@10 ~= 37/40
```

Если качество сильно отличается, проверить:

```text
model
query_prefix
document_prefix
collection
cache_path
providers
device_ids
```

---

# Проверить collection в Qdrant

```bash
python - <<'PY'
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

for name in [
    "bitrix_code_mvp_multilingual_e5_large",
    "bitrix_code_mvp_multilingual_e5_large_cuda3060",
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

---

# Быстрое переключение профилей

## Fast default CPU

Использует обычные `configs/`:

```bash
uv run bitrix-rag search "getRows" \
  --source project_local \
  --mode qdrant-hybrid \
  --limit 5
```

## CPU E5 quality profile

```bash
uv run bitrix-rag search "getRows" \
  --source project_local \
  --mode qdrant-hybrid \
  --limit 5 \
  --config-dir .indexer/experiments/multilingual-e5-large/configs
```

## NVIDIA CUDA E5 profile

Из `.venv-cuda`:

```bash
bitrix-rag search "getRows" \
  --source project_local \
  --mode qdrant-hybrid \
  --limit 5 \
  --config-dir .indexer/experiments/multilingual-e5-large-cuda3060/configs
```

---

# Как сделать CUDA profile дефолтом

Есть два варианта.

## Вариант A. Рекомендованный: не менять `configs/`

Оставить default CPU в `configs/`.

Для CUDA всегда запускать с:

```bash
--config-dir .indexer/experiments/multilingual-e5-large-cuda3060/configs
```

Плюсы:

```text
не ломает AMD/CPU
не ломает обычную .venv
не смешивает CUDA и CPU окружения
явно видно, какой профиль используется
```

Это лучший вариант для MVP.

---

## Вариант B. Сделать CUDA default в основной конфигурации

Использовать только на NVIDIA-машине и только в `.venv-cuda`.

Изменить:

```text
configs/embeddings.yaml
```

На:

```yaml
dense:
  provider: fastembed
  model: intfloat/multilingual-e5-large
  batch_size: 8
  cache_enabled: true
  cache_path: .indexer/cache/embeddings_cuda3060.sqlite
  query_prefix: 'query: '
  document_prefix: 'passage: '
  cuda: true
  providers:
    - CUDAExecutionProvider
  device_ids:
    - 0

sparse:
  enabled: true
  model: Qdrant/bm25
```

Изменить:

```text
configs/qdrant.yaml
```

На:

```yaml
url: http://localhost:6333
collection: bitrix_code_mvp_multilingual_e5_large_cuda3060
dense_vector_name: dense
sparse_vector_name: sparse
distance: Cosine
```

После этого обычные команды будут использовать CUDA profile:

```bash
bitrix-rag search "getRows" \
  --source project_local \
  --mode qdrant-hybrid \
  --limit 5
```

И eval:

```bash
bitrix-rag eval --profile mvp --mode qdrant-hybrid
```

Минусы:

```text
на AMD/CPU машине этот default config будет неудобен;
нужна .venv-cuda;
если CUDA provider не подхватится, поиск/индексация упадут;
можно случайно начать писать не туда.
```

Поэтому для репозитория лучше не коммитить CUDA как общий default.

---

# Что коммитить

Коммитить можно:

```text
src/bitrix_rag_indexer/embeddings/dense.py
src/bitrix_rag_indexer/experiments/prepare.py
tests/test_dense_embedder_gpu_config.py
tests/test_prepare_dense_experiment.py
```

Не коммитить:

```text
.venv/
.venv-cuda/
.indexer/cache/
.indexer/experiments/* как обязательные runtime artefacts
Qdrant storage
downloaded models
```

Если нужно сохранить команды — добавить markdown report, например:

```text
eval/reports/cuda-e5-large-setup.md
```

---

# Что делать после CUDA benchmark

Если CUDA E5 быстрее и качество совпадает:

```text
E5 CUDA = quality profile for NVIDIA laptop
E5 CPU = quality profile for AMD desktop
bge-small CPU = fast default
```

Если CUDA не быстрее или постоянно OOM:

```text
не мучить MVP;
оставить E5 CPU как quality profile;
ускорять индексацию через:
  - cache;
  - fewer chunks;
  - filtering noisy files;
  - not reindexing unchanged files;
  - source-level indexing;
  - later separate sentence-transformers/PyTorch backend if needed.
```
