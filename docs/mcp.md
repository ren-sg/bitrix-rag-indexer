# Bitrix RAG MCP — Руководство

Встроенный сервер Model Context Protocol (MCP) обеспечивает бесшовную интеграцию вашей кодовой базы на Bitrix с современными AI-ассистентами (Claude Code, Cursor, Windsurf, Aider, Cherry Studio и др.).

---

## 🛠 Доступные инструменты (MCP Tools)

Для решения частой проблемы "переполнения контекста" (context flooding) у AI-моделей, сервер реализует паттерн **Multi-Tool Architecture**. Он предоставляет два разных инструмента, чтобы и человек, и агент могли выбирать оптимальный способ работы с кодовой базой.

### 1. `bitrix_code_locator` (Инструмент разведки и навигации)
**Назначение**: Широкий поиск по проекту для понимания архитектуры и нахождения путей к файлам.
* **Особенность**: Возвращает **только метаданные** (пути, номера строк, сигнатуры методов), без огромных кусков исходного кода. Очень экономит токены (token-efficient). При `USE_ABS_PATH=true` добавляет `abs_path` рядом с `rel_path`.
* **Когда использовать агенту**: При вопросах вида *"Найди все места, где вызывается `BX.ajax`"*, *"Где лежит логика оформления заказа?"*, *"Покажи структуру модуля `im`"*.
* **Пример работы**:
  Агент запрашивает: `bitrix_code_locator(query="класс корзины", limit=10)`
  Получает список путей (например, `rel_path: bitrix/modules/sale/lib/discountcouponsmanagerbase.php`, `abs_path: /home/user/www/bitrix/modules/sale/lib/discountcouponsmanagerbase.php`, строки 150-300). После этого агент может использовать свой собственный инструмент (например, `read_file` в Claude Code), чтобы прочитать нужные строки.

### 2. `bitrix_semantic_search` (Инструмент точных решений)
**Назначение**: Получение конкретных реализаций и готового кода.
* **Особенность**: Возвращает **полный текст** найденных фрагментов кода, а также `rel_path` и (при `USE_ABS_PATH=true`) `abs_path`. По умолчанию лимит снижен, чтобы выдавать только самые релевантные куски.
* **Когда использовать агенту**: При вопросах вида *"Покажи реализацию метода `getRows`"*, *"Как в этом проекте сохраняются пользователи?"*, *"Найди ошибку в функции `processPayment`"*.
* **Пример работы**:
  Агент запрашивает: `bitrix_semantic_search(query="реализация getRows", limit=2)`
  Сразу получает текст кода метода и пишет ответ пользователю.

### 3. Доступные фильтры для агентов
Оба инструмента (`bitrix_code_locator` и `bitrix_semantic_search`) поддерживают мощные фильтры, которые агент может комбинировать:
* `limit` (int): Максимальное количество результатов.
* `lang` (str): Язык программирования (по умолчанию `"php"`, но можно передать `"javascript"`, `"vue"` и т.д.).
* `project` (str): Поиск строго по названию проекта (например, `"example_project"`). По умолчанию ищет по всем проектам.
* `path` (str): Поиск по подстроке `rel_path` в payload (например, `"local/components"`, `"bitrix/modules/sale"`).
* `php_namespace` (str): Строгий поиск внутри PHP namespace (например, `"Bitrix\Sale"`). Работает благодаря встроенному парсеру Tree-Sitter.
* `php_class` (str): Строгий поиск только внутри конкретного PHP класса или интерфейса (например, `"Basket"`). Полезно для поиска конкретных методов внутри гигантских классов.

### 4. Поля ответа поиска

Каждый результат содержит:
* `rel_path` — путь относительно `project.root` (с префиксом `path`, если задан в конфиге проекта)
* `abs_path` — абсолютный путь на диске (только при `USE_ABS_PATH=true`)
* `project`, `language`, `start_line`, `end_line` — метаданные чанка
* `path` — legacy alias для `rel_path`

`bitrix_code_locator` дополнительно возвращает `signature` и `symbol` вместо полного текста кода.

---

## 🚀 1. Запуск сервера

MCP сервер запускается в Docker-окружении вместе с базой Qdrant.

```bash
# Поднимаем Qdrant (если еще не поднят)
docker compose up -d qdrant

# Запускаем MCP сервер
docker compose up -d bitrix-rag-mcp
```

Для проверки здоровья сервера (Healthcheck):
```bash
curl http://localhost:8000/healthz
# Ожидаемый ответ: {"status": "ok", "service": "bitrix-rag-indexer-mcp"}
```

---

## ⚙️ 2. Конфигурация и кэш

### 2.1 Переменные окружения
Основной путь конфигурации задается в `.env`. По умолчанию это директория `configs/`:
```env
BITRIX_RAG_CONFIG_DIR_HOST=./configs
BITRIX_MODULES_ROOT=/home/user/www/
USE_ABS_PATH=true
```

`USE_ABS_PATH=true` включает поле `abs_path` в результатах поиска (MCP и CLI). Абсолютный путь вычисляется из `project.root` и конфигурации проекта:
- при `force_rel_path: true` (миграция старых данных): `project.root / path / stored_rel_path`
- иначе: `project.root / rel_path`

CLI-поиск (`uv run bitrix-rag search`) использует ту же логику через общий middleware.

Шаблон локального конфига проекта: `configs/projects/my_project.local.yaml.example`.

#### Временная миграция: `force_rel_path`

Если данные были проиндексированы с неправильным `root` (без префикса `path` в `rel_path`), добавьте в конфиг проекта:

```yaml
path: bitrix/modules
force_rel_path: true
```

Middleware добавит префикс `path` к `rel_path` при выдаче без переиндексации. После переиндексации с корректным конфигом — удалите `force_rel_path`.

*Если вы изменили `.env`, необходимо пересоздать контейнер:*
```bash
docker compose up -d --force-recreate bitrix-rag-mcp
```

### 2.2 Проблема с Read-Only БД (Кэш)
Для ускорения работы используется SQLite кэш эмбеддингов. Если вы получаете ошибку `OperationalError: attempt to write a readonly database`, обновите права локально:
```bash
sudo chown -R $USER:$USER .indexer/cache
```

---

## 🤖 3. Настройка AI-клиента (например, Cherry Studio)

В большинстве умных агентов (Claude Code, Cursor) MCP инструменты работают из коробки благодаря мощным описаниям (docstrings) внутри самого кода.

Если вы используете клиенты, требующие системного промпта, добавьте следующие правила для модели:
```text
- Всегда используй MCP инструменты для поиска по кодовой базе.
- Если нужно найти путь или архитектуру — используй bitrix_code_locator.
- Если нужна реализация метода — используй bitrix_semantic_search с limit=2.
- Не пытайся генерировать SQL-запросы Bitrix без предварительного поиска примеров.
```

---

## 📋 4. Наблюдение и отладка

**Чтение логов сервера:**
```bash
docker logs -f bitrix-rag-mcp
```

**Проверка подключенных томов конфигурации (внутри Docker):**
```bash
docker inspect bitrix-rag-mcp --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
```

**Отладка выбора модели внутри контейнера:**
```bash
docker exec -i bitrix-rag-mcp python - <<'PY'
from pathlib import Path
import yaml
data = yaml.safe_load(Path("/app/configs/embeddings.yaml").read_text())
print(data["dense"]["model"])
PY
```

**CLI-поиск с путями:**
```bash
USE_ABS_PATH=true uv run bitrix-rag search "DiscountCouponsManager" --project bitrix_modules -n 3
```

---

## 🧹 5. Минимальный чек-лист перед работой

```bash
git status
uv run pytest -q
docker compose ps
curl http://localhost:8000/healthz
```
