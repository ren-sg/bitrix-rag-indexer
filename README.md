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

**Индексация каталога `local` в битрикс окружении**
```bash
uv run bitrix-rag index --profile mvp --source project_local
```
