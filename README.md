#### Команды, которые актуальны

> ⚠️ **После обновления нужен полный переиндекс** — удалите коллекцию Qdrant и файл
> `.indexer/state/index.sqlite` перед первым запуском.

---

**Настройка:**

1. Скопируйте `.env.example` в `.env` и укажите пути к проектам:

```env
MY_PROJECT_ROOT=/absolute/path/to/my_project
```

2. Создайте конфиг проекта в `configs/projects/my_project.local.yaml`:

```yaml
project: my_project
root: ${MY_PROJECT_ROOT}
path: local                      # поддиректория root (опционально)

include:
  - "**/*.php"
  - "**/*.js"
  - "**/*.ts"
  - "**/*.vue"
  - "**/*.md"

exclude_from:
  - "excludes.bitrix.yaml"      # общие Bitrix-паттерны

exclude:
  - "**/my_custom_exclude/**"
```

---

**Проверка:**

```bash
uv run pytest -q
```

**Индексация:**

```bash
uv sync
docker compose up -d qdrant

# Индексировать все проекты из configs/projects/
uv run bitrix-rag index

# Только один проект
uv run bitrix-rag index --project my_project

# Только PHP-файлы
uv run bitrix-rag index --project my_project --lang php

# Dry-run (без записи)
uv run bitrix-rag index --project my_project --dry-run

# Принудительный переиндекс
uv run bitrix-rag index --project my_project --force

# Ограничить количество файлов
uv run bitrix-rag index --project my_project --max-files 30 --force
```

**Prune:**

```bash
uv run bitrix-rag prune --project my_project --dry-run
uv run bitrix-rag prune --project my_project
```

**Search:**

```bash
uv run bitrix-rag search "BX.ajax" --project my_project --mode dense --limit 10
uv run bitrix-rag search "BX.ajax" --project my_project --mode qdrant-sparse
uv run bitrix-rag search "BX.ajax" --project my_project --mode qdrant-hybrid

uv run bitrix-rag search "getRows" \
  --project my_project \
  --lang php \
  --mode qdrant-hybrid \
  --limit 5 \
  --debug
```

**Eval:**

```bash
# Использует eval/queries.local.yaml или eval/queries.yaml
uv run bitrix-rag eval --mode dense
uv run bitrix-rag eval --mode qdrant-sparse
uv run bitrix-rag eval --mode qdrant-hybrid

# Указать файл явно
uv run bitrix-rag eval --file eval/queries.my_project.yaml --mode qdrant-hybrid

# Короткий summary
uv run bitrix-rag eval --mode qdrant-hybrid | grep 'Summary'
```
