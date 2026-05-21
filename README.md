# Bitrix RAG Indexer

**Bitrix RAG Indexer** — это высокоэффективный инструмент для индексации исходного кода проектов на базе Bitrix CMS и подготовки данных для RAG (Retrieval-Augmented Generation). 

Инструмент сканирует файлы проектов (PHP, JS, TS, Vue, Markdown, TXT), разбивает их на смысловые фрагменты (чанки) с использованием синтаксического анализа (AST Tree-Sitter) для PHP/кода и сохраняет векторные представления в Qdrant. 

Также проект предоставляет встроенный **MCP (Model Context Protocol) сервер**, который позволяет искусственному интеллекту (LLM, AI-агентам в IDE/клиентах вроде Cursor, Windsurf, Cherry Studio и др.) осуществлять точный гибридный поиск по вашей кодовой базе непосредственно во время разработки.

---

## 🚀 Основные возможности

* 🌳 **Умный чанкинг на основе AST (Tree-Sitter)**: Для PHP (и стандартного кода) используется разбор синтаксического дерева, что позволяет сохранять границы классов, методов и функций вместе с их контекстом, в отличие от простого текстового разделения.
* 🔍 **Гибридный поиск (Hybrid Search)**:
  * **Dense (векторный)**: семантический поиск по смыслу (использует модели `fastembed`, такие как `intfloat/multilingual-e5-large` или `BAAI/bge-small-en-v1.5`).
  * **Sparse (разреженный)**: поиск точных совпадений ключевых слов, символов, функций и классов (использует разреженные векторы Qdrant Splade/BM25).
  * Объединение результатов в режиме `qdrant-hybrid` обеспечивает наивысшую точность поиска по кодовой базе.
* ⚡ **Инкрементальная индексация (Incremental Indexing)**:
  * Локальное хранение состояния в SQLite (`.indexer/state/index.sqlite`) позволяет отслеживать хэши файлов и индексировать только изменившиеся файлы.
  * Поддержка команды `prune` для автоматического удаления из векторной базы файлов, удаленных из физического проекта.
* 🔌 **Model Context Protocol (MCP)**:
  * Встроенный MCP-сервер позволяет бесшовно интегрировать вашу кодовую базу с любыми современными AI-ассистентами.
* 📂 **Полная переносимость данных**:
  * Все данные индекса (база SQLite, кэши эмбеддингов, векторное хранилище Qdrant) находятся внутри единого изолированного каталога `.indexer/`. Эту папку легко скопировать/архивировать для мгновенного переноса готового индекса на другой компьютер без повторной индексации.

---

## ⚙️ Настройка

> ⚠️ **Важное примечание:** После обновления структуры индексатора может потребоваться полный переиндекс. Для этого удалите коллекцию Qdrant и файл `.indexer/state/index.sqlite` перед первым запуском.

1. **Настройка окружения:**
   Скопируйте шаблон переменных окружения `.env.example` в `.env` и укажите абсолютный путь к вашему проекту Bitrix:
   ```env
   MY_PROJECT_ROOT=/absolute/path/to/my_project
   ```

2. **Создание конфигурации проекта:**
   Создайте конфигурационный YAML-файл для вашего проекта в папке `configs/projects/my_project.local.yaml`:
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
     - "excludes.bitrix.yaml"      # общие Bitrix-паттерны исключений
   
   exclude:
     - "**/my_custom_exclude/**"
   ```

---

## 🧪 Проверка (Тесты)

Для запуска тестов проекта выполните:
```bash
uv run pytest -q
```

---

## 🔨 Индексация кодовой базы

Сначала установите зависимости проекта и поднимите контейнер Qdrant:
```bash
uv sync
docker compose up -d qdrant
```

### Команды индексации:

```bash
# Индексировать все проекты, настроенные в configs/projects/
uv run bitrix-rag index

# Индексировать только один конкретный проект
uv run bitrix-rag index --project my_project

# Индексировать в проекте только PHP-файлы
uv run bitrix-rag index --project my_project --lang php

# Dry-run (тестовый запуск без записи изменений в базу данных)
uv run bitrix-rag index --project my_project --dry-run

# Принудительный полный переиндекс (без учета хэшей)
uv run bitrix-rag index --project my_project --force

# Ограничить количество индексируемых файлов за запуск
uv run bitrix-rag index --project my_project --max-files 30 --force
```

---

## 🧹 Очистка индекса (Prune)

Используйте команду `prune` для удаления из поискового индекса и Qdrant данных о файлах, которые были физически удалены из кодовой базы проекта:

```bash
# Проверить, какие данные будут удалены (тестовый режим)
uv run bitrix-rag prune --project my_project --dry-run

# Применить очистку индекса
uv run bitrix-rag prune --project my_project
```

---

## 🔍 Поиск по индексу (CLI)

Вы можете протестировать качество работы поиска прямо из командной строки в различных режимах:

```bash
# Поиск в векторном (семантическом) режиме (Dense)
uv run bitrix-rag search "BX.ajax" --project my_project --mode dense --limit 10

# Поиск в разреженном полнотекстовом режиме (Sparse)
uv run bitrix-rag search "BX.ajax" --project my_project --mode qdrant-sparse

# Поиск в гибридном режиме (Dense + Sparse)
uv run bitrix-rag search "BX.ajax" --project my_project --mode qdrant-hybrid

# Отладка подробного вывода при гибридном поиске в PHP-файлах
uv run bitrix-rag search "getRows" \
  --project my_project \
  --lang php \
  --mode qdrant-hybrid \
  --limit 5 \
  --debug
```

---

## 📈 Оценка качества поиска (Eval)

Для оценки эффективности различных алгоритмов поиска и моделей эмбеддингов используется тестовая выборка запросов:

```bash
# Запуск оценки на основе файла eval/queries.local.yaml или eval/queries.yaml
uv run bitrix-rag eval --mode dense
uv run bitrix-rag eval --mode qdrant-sparse
uv run bitrix-rag eval --mode qdrant-hybrid

# Запуск оценки с указанием конкретного файла тестовых запросов
uv run bitrix-rag eval --file eval/queries.my_project.yaml --mode qdrant-hybrid

# Вывод только краткого Summary результатов оценки
uv run bitrix-rag eval --mode qdrant-hybrid | grep 'Summary'
```

---

## ⚡ Запуск с поддержкой CUDA (GPU-ускорение)

Для ускорения процесса генерации эмбеддингов на видеокартах NVIDIA используйте специальное виртуальное окружение:

```bash
source .venv-cuda/bin/activate
source scripts/env_cuda.sh
bitrix-rag index --project my_project --dry-run
```

---

## 📁 Структура директории `.indexer/` (Переносимый кэш)

Каталог `.indexer/` содержит все необходимые данные вашего RAG-окружения:
* **`.indexer/qdrant_storage/`**: Вся векторная база данных Qdrant (коллекции, векторы, пэйлоады).
* **`.indexer/state/index.sqlite`**: Локальная база данных SQLite, хранящая хэши проиндексированных файлов и структуру чанков.
* **`.indexer/cache/`**: Кэшированные эмбеддинги для предотвращения повторных вызовов моделей при неизменном тексте чанков.
* **`.indexer/experiments/`**: Каталог для проведения экспериментов с различными конфигурациями.

Для переноса всего проиндексированного состояния на другой компьютер достаточно передать эту папку целиком.
