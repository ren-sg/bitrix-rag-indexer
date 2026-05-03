# NVIDIA CUDA profile for FastEmbed / E5

## Цель

Добавить отдельный CUDA-профиль для `intfloat/multilingual-e5-large`, чтобы ускорить dense embedding на NVIDIA GPU и не ломать обычную CPU-индексацию.

Текущая стратегия MVP:

```text
Fast default:
  BAAI/bge-small-en-v1.5
  CPU
  qdrant-hybrid

Quality CPU:
  intfloat/multilingual-e5-large
  CPU
  qdrant-hybrid

Quality NVIDIA:
  intfloat/multilingual-e5-large
  CUDA
  qdrant-hybrid
```

Основное правило:

> CUDA-профиль держать отдельно: отдельная `.venv-cuda`, отдельная Qdrant collection, отдельный embedding cache.

Это нужно, чтобы не смешивать CPU/GPU окружения и не ломать уже рабочие collections.

Архитектурно это соответствует текущей цели MVP: улучшать качество и измеримость индексации через chunking, metadata, dense/sparse/hybrid retrieval и eval, а не ломать общий pipeline. :contentReference[oaicite:0]{index=0}

---

# 1. Окружения

## Обычное CPU окружение

Используется для AMD desktop, обычной разработки и стабильного MVP.

```text
.venv
```

Внутри:

```text
fastembed
onnxruntime
```

Проверка:

```bash
uv run pytest -q
```

Ожидаемо:

```text
28 passed
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

Нормально:

```text
ORT providers: ['AzureExecutionProvider', 'CPUExecutionProvider']
FastEmbed import: OK
```

## NVIDIA CUDA окружение

Используется только для CUDA-профиля.

```text
.venv-cuda
```

Внутри:

```text
fastembed-gpu
onnxruntime-gpu
CUDA runtime libraries from NVIDIA wheels
```

Не смешивать `.venv` и `.venv-cuda`.

---

# 2. AMD / Radeon

Для AMD GPU в текущем MVP ничего не трогаем.

На AMD использовать только CPU-режим:

```bash
rm -rf .venv
uv venv .venv --python 3.12
uv sync
uv run pytest -q
```

На AMD не ставить в основную `.venv`:

```text
fastembed-gpu
onnxruntime-gpu
```

Причина: текущий FastEmbed GPU-путь практически ориентирован на CUDA/NVIDIA. Для AMD нужен отдельный backend позже:

```text
sentence-transformers + PyTorch ROCm
или отдельный ONNX Runtime / MIGraphX backend
```

Это не часть текущего MVP.

---

# 3. Создать NVIDIA CUDA venv

Из корня проекта:

```bash
deactivate 2>/dev/null || true

uv python install 3.12
uv venv .venv-cuda --python 3.12

source .venv-cuda/bin/activate
```

Проверить:

```bash
python -V
which python
```

Ожидаемо:

```text
Python 3.12.x
.../bitrix-rag-indexer/.venv-cuda/bin/python
```

---

# 4. Установить проект и FastEmbed GPU

В активированной `.venv-cuda`:

```bash
uv pip install -e .
uv pip uninstall fastembed onnxruntime
uv pip install fastembed-gpu
```

Важно: у `uv pip uninstall` нет флага `-y`.

Проверить пакеты:

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

Для CUDA-окружения нормально:

```text
fastembed not installed
fastembed-gpu installed
onnxruntime not installed
onnxruntime-gpu installed
```

---

# 5. Проверить ONNX Runtime CUDA provider

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

Пример рабочего результата:

```text
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

Если видишь только:

```text
['AzureExecutionProvider', 'CPUExecutionProvider']
```

CUDA backend не подхватился. Индексацию CUDA collection не запускать.

---

# 6. CUDA runtime libraries через NVIDIA wheels

Если CUDA provider виден, но при создании модели появляется ошибка вида:

```text
libcublasLt.so.12: cannot open shared object file
libcurand.so.10: cannot open shared object file
libcufft.so.11: cannot open shared object file
```

нужно поставить CUDA runtime libraries в `.venv-cuda`.

Минимальный набор, который понадобился для текущего запуска:

```bash
uv pip install \
  nvidia-cublas-cu12 \
  nvidia-cuda-runtime-cu12 \
  nvidia-cudnn-cu12 \
  nvidia-curand-cu12 \
  nvidia-cufft-cu12
```

Выставить `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$PWD/.venv-cuda/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.venv-cuda/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$PWD/.venv-cuda/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.venv-cuda/lib/python3.12/site-packages/nvidia/curand/lib:$PWD/.venv-cuda/lib/python3.12/site-packages/nvidia/cufft/lib:${LD_LIBRARY_PATH:-}"
```

Проверить, что библиотеки находятся:

```bash
python - <<'PY'
import site
from pathlib import Path

for root in site.getsitepackages():
    root_path = Path(root) / "nvidia"
    if root_path.exists():
        for pattern in [
            "libcublasLt.so*",
            "libcudart.so*",
            "libcudnn.so*",
            "libcurand.so*",
            "libcufft.so*",
        ]:
            for path in root_path.rglob(pattern):
                print(path)
PY
```

---

# 7. Проверить FastEmbed CUDA

Важно: использовать `cuda=True`, а не одновременно `cuda=True` и `providers=["CUDAExecutionProvider"]`.

```bash
python - <<'PY'
from fastembed import TextEmbedding

model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    cuda=True,
    device_ids=[0],
)

print(model.model.model.get_providers())
PY
```

Ожидаемо:

```text
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Если вывод:

```text
['CPUExecutionProvider']
```

CUDA не используется.

---

# 8. Создать CUDA experiment config

Создать отдельную collection и отдельный cache:

```bash
python -m bitrix_rag_indexer.experiments.prepare \
  --name multilingual-e5-large-cuda3060 \
  --model "intfloat/multilingual-e5-large" \
  --collection "bitrix_code_mvp_multilingual_e5_large_cuda3060" \
  --query-prefix "query: " \
  --document-prefix "passage: " \
  --cache-path ".indexer/cache/embeddings_cuda3060.sqlite" \
  --cuda \
  --device-id 0 \
  --overwrite
```

Важно: не передавать `--provider "CUDAExecutionProvider"` вместе с `--cuda`.

Если generator уже создал `providers`, убрать его вручную из generated config.

---

# 9. Проверить generated config

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

Если RTX 3060 6GB падает по OOM, снизить:

```yaml
batch_size: 32
```

до:

```yaml
batch_size: 8
```

или:

```yaml
batch_size: 4
```

В текущем прогоне `batch_size: 32` отработал успешно.

---

# 10. Проверить DenseEmbedder проекта

```bash
python - <<'PY'
from pathlib import Path
import yaml

from bitrix_rag_indexer.embeddings.dense import DenseEmbedder

config_path = Path(".indexer/experiments/multilingual-e5-large-cuda3060/configs/embeddings.yaml")

with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

dense_cfg = cfg["dense"]
print("dense config:", dense_cfg)

embedder = DenseEmbedder(dense_cfg)
print("embedder cuda:", getattr(embedder, "cuda", None))
print("embedder providers:", getattr(embedder, "providers", None))
print("embedder device_ids:", getattr(embedder, "device_ids", None))
print("onnx providers:", embedder._model.model.model.get_providers())
PY
```

Нужно увидеть:

```text
embedder cuda: True
embedder providers: None
embedder device_ids: [0]
onnx providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Если видишь:

```text
onnx providers: ['CPUExecutionProvider']
```

индексацию не запускать.

---

# 11. Индексация CUDA collection

Запустить Qdrant:

```bash
docker compose up -d qdrant
```

В активированной `.venv-cuda` и с выставленным `LD_LIBRARY_PATH`:

```bash
bitrix-rag index \
  --profile mvp \
  --source project_local \
  --force \
  --config-dir .indexer/experiments/multilingual-e5-large-cuda3060/configs
```

Или явно через Python из `.venv-cuda`:

```bash
python -m bitrix_rag_indexer.cli index \
  --profile mvp \
  --source project_local \
  --force \
  --config-dir .indexer/experiments/multilingual-e5-large-cuda3060/configs
```

Не использовать `uv run` для CUDA-профиля, пока есть отдельная `.venv-cuda`, чтобы случайно не запустить обычную `.venv`.

---

# 12. Мониторинг GPU

В другом терминале:

```bash
watch -n 0.5 nvidia-smi
```

CPU всё равно будет загружен частично:

```text
tokenization
batching
file reading
payload build
Qdrant upsert
SQLite manifest
sparse/BM25
```

Но если CUDA используется, VRAM должна быть занята, а `dense_embed` должен быть сильно быстрее CPU.

---

# 13. Текущий результат CUDA E5

CPU E5:

```text
dense_embed: 2736.61s
total: 2907.01s
RSS: 3553 MB
```

CUDA E5 on RTX 3060 6GB:

```text
files scanned: 573
files indexed: 573
chunks created: 3659
scanned_mb: 3.3

dense_embed: 145.18s
qdrant_upsert: 6.30s
manifest_replace: 5.20s
total: 160.75s
RSS: 1244 MB
```

Ускорение:

```text
dense_embed: примерно 18.8x быстрее
total: примерно 18.1x быстрее
```

---

# 14. Eval CUDA collection

```bash
bitrix-rag eval --profile mvp --mode dense --config-dir .indexer/experiments/multilingual-e5-large-cuda3060/configs | grep 'Summary\|Path-only'
bitrix-rag eval --profile mvp --mode qdrant-sparse --config-dir .indexer/experiments/multilingual-e5-large-cuda3060/configs | grep 'Summary\|Path-only'
bitrix-rag eval --profile mvp --mode qdrant-hybrid --config-dir .indexer/experiments/multilingual-e5-large-cuda3060/configs | grep 'Summary\|Path-only'
```

Ожидаемо качество примерно как у CPU E5:

```text
dense:
hit@10 ~= 35/40

qdrant-hybrid:
hit@10 ~= 37/40
path@10 ~= 37/40
```

CPU E5 baseline:

```text
dense:
Summary: total=40, hit@5=34/40 (85%), hit@10=35/40 (88%)
Path-only Summary: total=40, path_hit@5=35/40 (88%), path_hit@10=36/40 (90%)

qdrant-hybrid:
Summary: total=40, hit@5=37/40 (92%), hit@10=37/40 (92%)
Path-only Summary: total=40, path_hit@5=37/40 (92%), path_hit@10=37/40 (92%)
```

Если CUDA eval сильно отличается, проверить:

```text
model
query_prefix
document_prefix
collection
cache_path
cuda
device_ids
batch_size
```

---

# 15. Проверить Qdrant collections

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

# 16. Warnings

## Mean pooling warning

```text
The model intfloat/multilingual-e5-large now uses mean pooling instead of CLS embedding.
```

Это не ошибка. Это предупреждение FastEmbed о текущем pooling behavior. E5 дал лучший результат на eval, поэтому старую версию FastEmbed сейчас не пинним.

## CUDA/providers warning

```text
`cuda` and `providers` are mutually exclusive parameters
```

Это нужно исправить в config.

Неправильно:

```yaml
cuda: true
providers:
  - CUDAExecutionProvider
```

Правильно:

```yaml
cuda: true
device_ids:
  - 0
```

## ONNX nodes assigned to CPU

```text
Some nodes were not assigned to the preferred execution providers...
ORT explicitly assigns shape related ops to CPU to improve perf.
```

Это не ошибка. ONNX Runtime может оставлять shape-related операции на CPU. Если при этом `dense_embed` ускорился и VRAM используется, CUDA работает.

---

# 17. Быстрое переключение профилей

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

## NVIDIA CUDA E5 quality profile

Из `.venv-cuda`:

```bash
source .venv-cuda/bin/activate

export LD_LIBRARY_PATH="$PWD/.venv-cuda/lib/python3.12/site-packages/nvidia/cublas/lib:$PWD/.venv-cuda/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$PWD/.venv-cuda/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/.venv-cuda/lib/python3.12/site-packages/nvidia/curand/lib:$PWD/.venv-cuda/lib/python3.12/site-packages/nvidia/cufft/lib:${LD_LIBRARY_PATH:-}"

bitrix-rag search "getRows" \
  --source project_local \
  --mode qdrant-hybrid \
  --limit 5 \
  --config-dir .indexer/experiments/multilingual-e5-large-cuda3060/configs
```

---

# 18. Как сделать CUDA профилем по умолчанию

## Вариант A. Рекомендованный

Не менять `configs/`.

Для CUDA всегда использовать:

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

## Вариант B. Сделать CUDA default в `configs/`

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
  batch_size: 32
  cache_enabled: true
  cache_path: .indexer/cache/embeddings_cuda3060.sqlite
  query_prefix: 'query: '
  document_prefix: 'passage: '
  cuda: true
  device_ids:
    - 0

sparse:
  enabled: true
  model: Qdrant/bm25
```

Если будет OOM:

```yaml
batch_size: 8
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

После этого обычные команды в `.venv-cuda` будут использовать CUDA profile:

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
на AMD/CPU машине такой default неудобен;
нужна .venv-cuda;
нужен LD_LIBRARY_PATH;
если CUDA provider не подхватится, поиск/индексация упадут или уйдут в CPU;
можно случайно начать писать не в ту collection.
```

Для общего репозитория лучше не коммитить CUDA как default.

---

# 19. Что коммитить

Коммитить можно:

```text
src/bitrix_rag_indexer/embeddings/dense.py
src/bitrix_rag_indexer/experiments/prepare.py
tests/test_dense_embedder_gpu_config.py
tests/test_prepare_dense_experiment.py
eval/reports/cuda-e5-large-setup.md
```

Не коммитить:

```text
.venv/
.venv-cuda/
.indexer/cache/
.indexer/experiments/
Qdrant storage
downloaded models
NVIDIA wheel binaries
```

---

# 20. Что делать дальше после CUDA benchmark

Если CUDA E5 стабильно работает:

```text
E5 CUDA = quality profile for NVIDIA laptop
E5 CPU = quality profile for AMD desktop
bge-small CPU = fast default
```

Дальше модельные эксперименты остановить.

Следующие улучшения делать не через выбор модели, а через качество корпуса:

```text
1. noisy files / большие JS и template blobs
2. failed cases по E5 hybrid
3. chunk count reduction
4. eval coverage
5. индексация bitrix/modules/im
6. отдельные source profiles
```
