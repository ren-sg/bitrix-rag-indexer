# Bitrix RAG MCP — быстрый старт

Минимальная инструкция: запуск, конфигурация, проверка, наблюдение.

---

## 1. Запуск

### 1.1 Поднять Qdrant

```bash
docker compose up -d qdrant
```

### 1.2 Запустить MCP сервер

```bash
docker compose up -d bitrix-rag-mcp
```

> Для смены конфига используем переменную окружения (см. ниже).

---

## 2. Конфигурация

### 2.1 Основная переменная

В `.env`:

```env
BITRIX_RAG_CONFIG_DIR_HOST=.indexer/experiments/multilingual-e5-large/configs
```

И перезапуск:

```bash
docker compose up -d --force-recreate bitrix-rag-mcp
```

---

### 2.2 Проверка, что конфиг применился

```bash
docker exec -i bitrix-rag-mcp python - <<'PY'
from pathlib import Path
print(Path("/app/configs/embeddings.yaml").read_text())
PY
```

Ожидаемо:

```yaml
model: intfloat/multilingual-e5-large
```

---

### 2.3 Кеш embeddings

```yaml
cache_path: .indexer/cache/embeddings.sqlite
```

Важно:

```bash
sudo chown -R $USER:$USER .indexer/cache
```

Иначе будет:

```text
OperationalError: attempt to write a readonly database
```

---

## 3. Проверка работы

### 3.1 Healthcheck

```bash
curl http://localhost:8000/healthz
```

Ожидаемо:

```text
200 OK
```

---

### 3.2 Прямой поиск (локально)

```bash
uv run bitrix-rag search "getRows" \
  --source project_local \
  --mode qdrant-hybrid \
  --limit 5 \
  --config-dir .indexer/experiments/multilingual-e5-large/configs
```

---

### 3.3 Проверка из контейнера

```bash
docker logs -f bitrix-rag-mcp
```

Должны быть:

```text
GET /healthz 200
search requests
```

---

## 4. Использование в MCP (например, Cherry Studio)

Минимальные правила для модели:

```text
- всегда использовать MCP для поиска кода
- если нет результатов — повторить поиск
- не использовать SQL без найденного примера
- использовать найденные чанки как основу
```

---

## 5. Наблюдение

### 5.1 Логи

```bash
docker logs -f bitrix-rag-mcp
```

---

### 5.2 Проверка подключённого конфига

```bash
docker inspect bitrix-rag-mcp --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
```

Ожидаемо:

```text
.indexer/.../configs -> /app/configs
.indexer/cache -> /app/.indexer/cache
```

---

### 5.3 Проверка модели (внутри контейнера)

```bash
docker exec -i bitrix-rag-mcp python - <<'PY'
from pathlib import Path
import yaml

data = yaml.safe_load(Path("/app/configs/embeddings.yaml").read_text())
print(data["dense"]["model"])
PY
```

---

## 6. Частые проблемы

### ❌ readonly sqlite

```bash
sudo chown -R $USER:$USER .indexer/cache
```

---

### ❌ конфиг не применяется

```bash
docker compose up -d --force-recreate bitrix-rag-mcp
```

---

### ❌ MCP "не ищет"

Проверить:

```text
- MCP реально вызывается (не "предположим")
- limit >= 5
- source указан
```

---

### ❌ таймауты

```text
- увеличить timeout в клиенте (Cherry)
- проверить, что Qdrant жив
```

---

## 7. Рекомендуемый профиль

```text
model: intfloat/multilingual-e5-large
mode: qdrant-hybrid
```

---

## 8. Минимальный чек перед работой

```bash
git status
uv run pytest -q
docker compose ps
curl http://localhost:8000/healthz
```

---

## Готово

MCP сервер поднят и готов к использованию.
