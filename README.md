#### Проверка

```bash
uv sync
docker compose up -d qdrant

uv run bitrix-rag index --profile mvp --source manual
uv run bitrix-rag stats
uv run bitrix-rag search "где описаны современные компоненты битрикс"
uv run bitrix-rag search "зачем нужен каталог local"
uv run bitrix-rag search "что делает модуль im"
```

**Ожидаемый результат:**

```
Indexed files=3, chunks=3
```


#### Запуск
**Проверка файлов для индексации в `local`**
```bash
uv run bitrix-rag index --profile mvp --source project_local --dry-run
```

**Индексация каталога `local` в битрикс окружении**
```bash
uv run bitrix-rag index --profile mvp --source project_local
```

#### Очистка и переиндексация

```bash
## Re-index
uv run bitrix-rag index --profile mvp --source project_local --force

## Clear
curl -X DELETE http://localhost:6333/collections/bitrix_code_mvp
rm -rf .indexer/state/index.sqlite
```

#### Очистка неиспользуемых данных в manifest
```bash
uv run bitrix-rag prune --profile mvp --source project_local --dry-run
uv run bitrix-rag prune --profile mvp --source project_local
```

#### Проверка eval

```bash
uv run bitrix-rag eval --profile mvp
```

```bash
expected:
  path_contains_any: []
  path_contains_all: []
  path_not_contains: []
  text_contains_any: []
  text_contains_all: []
  text_not_contains: []
```
#### Тесты

```bash
uv run pytest
```

#### Debug
```bash
uv run bitrix-rag search "BX.ajax" --mode hybrid --source project_local --limit 5 --debug
```
