#### Команды, которые актуальны

**Проверка:**

```bash
uv run pytest -q
```

**Индексация:**

```bash
uv sync
docker compose up -d qdrant

uv run bitrix-rag index --profile mvp --source manual
uv run bitrix-rag index --profile mvp --source project_local --dry-run
uv run bitrix-rag index --profile mvp --source project_local
uv run bitrix-rag index --profile mvp --source project_local --force
uv run bitrix-rag index --profile mvp --source project_local --max-files 30 --force
```

**Prune:**

```bash
uv run bitrix-rag prune --profile mvp --source project_local --dry-run
uv run bitrix-rag prune --profile mvp --source project_local
```

**Search:**

```bash
uv run bitrix-rag search "BX.ajax" --mode dense --source project_local --limit 10
uv run bitrix-rag search "BX.ajax" --mode qdrant-sparse --source project_local --limit 10
uv run bitrix-rag search "BX.ajax" --mode qdrant-hybrid --source project_local --limit 10

uv run bitrix-rag search "getRows" \
  --source project_local \
  --mode qdrant-hybrid \
  --limit 5 \
  --debug
```

**Eval:**

```bash
uv run bitrix-rag eval --profile mvp --mode dense
uv run bitrix-rag eval --profile mvp --mode qdrant-sparse
uv run bitrix-rag eval --profile mvp --mode qdrant-hybrid
```

**Короткий summary:**

```bash
uv run bitrix-rag eval --profile mvp --mode dense | grep 'Summary'
uv run bitrix-rag eval --profile mvp --mode qdrant-sparse | grep 'Summary'
uv run bitrix-rag eval --profile mvp --mode qdrant-hybrid | grep 'Summary'
```
