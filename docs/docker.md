# Docker — управление окружением

Проект использует Docker Compose для двух сервисов:

| Сервис | Контейнер | Порт | Назначение |
|---|---|---|---|
| `qdrant` | `bitrix-rag-qdrant` | `6333`, `6334` | Векторная база данных |
| `bitrix-rag-mcp` | `bitrix-rag-mcp` | `8000` | MCP-сервер поиска по коду |

Данные Qdrant и кэш эмбеддингов лежат на хосте в `.indexer/` и переживают пересоздание контейнеров.

---

## Что монтируется, а что «запекается» в образ

Это важно понимать, **когда** нужен `--build`, а когда достаточно `--force-recreate`.

| Изменение | Нужен `--build` | Нужен `--force-recreate` |
|---|---|---|
| Код в `src/` (MCP, поиск, фильтры) | **Да** | Да |
| `docker/mcp/Dockerfile`, `pyproject.toml`, `uv.lock` | **Да** | Да |
| `.env` (переменные окружения) | Нет | **Да** |
| `configs/` (YAML проектов, qdrant, embeddings) | Нет | Нет* |
| Данные в `.indexer/qdrant_storage/` | Нет | Нет |

\* `configs/` смонтированы read-only в контейнер. MCP подхватывает изменения конфигов при следующем запросе, **перезапуск обычно не нужен**. Исключение — если менялись переменные из `.env`, которые читаются только при старте контейнера.

**Типичная ошибка:** изменили Python-код, сделали только `docker compose up -d --force-recreate bitrix-rag-mcp` — контейнер пересоздался, но **образ остался старым**, агент продолжает ходить в старую версию API.

---

## Первый запуск

```bash
# из корня репозитория
cp .env.example .env
# отредактируйте .env: BITRIX_MODULES_ROOT, USE_ABS_PATH и т.д.

docker compose up -d
```

Поднимутся оба сервиса. Индексация выполняется **на хосте** через CLI (`uv run bitrix-rag index`), не внутри контейнера MCP.

---

## Основные команды

### Статус и логи

```bash
# список контейнеров и их состояние
docker compose ps

# логи MCP (follow)
docker logs -f bitrix-rag-mcp

# логи Qdrant
docker logs -f bitrix-rag-qdrant

# последние 100 строк MCP
docker logs --tail 100 bitrix-rag-mcp
```

### Запуск и остановка

```bash
# поднять всё
docker compose up -d

# только Qdrant (для индексации с хоста)
docker compose up -d qdrant

# только MCP
docker compose up -d bitrix-rag-mcp

# остановить без удаления контейнеров
docker compose stop

# остановить и удалить контейнеры (данные в .indexer/ сохраняются)
docker compose down
```

### Пересоздание и пересборка

```bash
# пересоздать контейнер MCP (новый .env, те же образ и код)
docker compose up -d --force-recreate bitrix-rag-mcp

# пересобрать образ и пересоздать контейнер (после изменений в src/)
docker compose up -d --build --force-recreate bitrix-rag-mcp

# пересобрать образ без запуска (проверка сборки)
docker compose build bitrix-rag-mcp

# пересоздать оба сервиса с пересборкой MCP
docker compose up -d --build --force-recreate
```

### Когда что делать

| Ситуация | Команда |
|---|---|
| Изменили `.env` (`USE_ABS_PATH`, `BITRIX_RAG_QDRANT_COLLECTION`, …) | `docker compose up -d --force-recreate bitrix-rag-mcp` |
| Изменили Python-код MCP / search / filters | `docker compose up -d --build --force-recreate bitrix-rag-mcp` |
| Обновили зависимости (`pyproject.toml`, `uv.lock`) | `docker compose build --no-cache bitrix-rag-mcp && docker compose up -d --force-recreate bitrix-rag-mcp` |
| Изменили `configs/projects/*.yaml` | перезапуск **не обязателен** (volume) |
| MCP «завис» или не отвечает | `docker compose restart bitrix-rag-mcp` |
| Cursor показывает старую схему MCP tools | rebuild + recreate MCP, затем переподключить MCP в IDE |
| Агент получает старый формат ответа (`filters` вместо `applied_filters`) | rebuild + recreate — в контейнере старый образ |

После пересборки MCP **переподключите MCP-сервер в Cursor** (Settings → MCP), иначе IDE может использовать закэшированную схему tools.

---

## Проверка версии кода внутри контейнера

Убедиться, что в контейнере актуальный код:

```bash
# не должно быть path: str в сигнатуре MCP tool
docker exec bitrix-rag-mcp grep -n "path: str" /app/src/bitrix_rag_indexer/mcp/server.py || echo "path filter removed OK"

# в search_service должен быть applied_filters
docker exec bitrix-rag-mcp grep -n "applied_filters" /app/src/bitrix_rag_indexer/mcp/search_service.py
```

---

## Права на кэш эмбеддингов

Если MCP падает с `OperationalError: attempt to write a readonly database`:

```bash
sudo chown -R $USER:$USER .indexer/cache
docker compose restart bitrix-rag-mcp
```

---

## Проверка через curl

Все команды ниже можно копировать и запускать в bash из корня репозитория (нужны запущенные контейнеры).

### Health / readiness MCP

```bash
curl -s http://localhost:8000/healthz
```

```bash
curl -s http://localhost:8000/readyz | python3 -m json.tool
```

Поле `"ready": true` означает, что можно вызывать MCP tools. Имя коллекции — в `stats.collection`.

### Qdrant

```bash
curl -s http://localhost:6333/collections | python3 -m json.tool
```

```bash
curl -s http://localhost:6333/collections/bitrix_code_mvp_e5_large | python3 -m json.tool
```

Подставьте имя коллекции из предыдущей команды или из `readyz` (`stats.collection`).

### MCP JSON-RPC (`http://localhost:8000/mcp`)

Каждый запрос — отдельная команда. Заголовки `Content-Type` и `Accept` обязательны.

#### Initialize

```bash
curl -s -X POST http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"curl","version":"1.0"}}}' \
  | python3 -m json.tool
```

#### Список tools

```bash
curl -s -X POST http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  | python3 -m json.tool
```

Проверьте, что в `inputSchema` **нет** устаревшего параметра `path`, если вы его убирали из кода.

#### Статистика индекса (`bitrix_code_stats`)

```bash
curl -s -X POST http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"bitrix_code_stats","arguments":{}}}' \
  | python3 -m json.tool
```

#### Поиск локаций кода (`bitrix_code_locator`)

```bash
curl -s -X POST http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"bitrix_code_locator","arguments":{"query":"create custom storage disk module","limit":3,"lang":"php","project":"bitrix_modules"}}}' \
  | python3 -m json.tool
```

Ожидаемый фрагмент в `result.structuredContent`:

```json
"applied_filters": {
  "project": "bitrix_modules",
  "lang": "php"
},
"count": 3
```

#### Семантический поиск с кодом (`bitrix_semantic_search`)

```bash
curl -s -X POST http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"bitrix_semantic_search","arguments":{"query":"Driver addStorage disk","limit":2,"lang":"php","project":"bitrix_modules","mode":"qdrant-hybrid"}}}' \
  | python3 -m json.tool
```

Результат содержит полный текст чанка в `results[].text` (может быть обрезан по `BITRIX_RAG_MAX_TEXT_CHARS`).

---

## Минимальный чек-лист после изменений

```bash
docker compose ps
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/readyz | python3 -m json.tool
docker exec bitrix-rag-mcp grep -c applied_filters /app/src/bitrix_rag_indexer/mcp/search_service.py
```

Если последняя команда возвращает `0` — образ не пересобран, выполните:

```bash
docker compose up -d --build --force-recreate bitrix-rag-mcp
```

---

## См. также

- [MCP — инструменты и настройка AI-клиента](./mcp.md)
- [README — индексация и CLI-поиск](../README.md)
