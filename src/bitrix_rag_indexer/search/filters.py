from dataclasses import dataclass

from qdrant_client import models

LANGUAGE_ALIASES = {
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "md": "markdown",
    "yml": "yaml",
}


def normalize_search_lang(lang: str | None) -> str | None:
    if lang is None:
        return None

    normalized = lang.strip().casefold()
    if not normalized:
        return None

    return LANGUAGE_ALIASES.get(normalized, normalized)


@dataclass(frozen=True)
class SearchFilters:
    project: str | None = None
    lang: str | None = None
    path: str | None = None
    php_namespace: str | None = None
    php_class: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "lang", normalize_search_lang(self.lang))

    def is_empty(self) -> bool:
        return not any(
            [
                self.project,
                self.lang,
                self.path,
                self.php_namespace,
                self.php_class,
            ]
        )


def optional_filter_value(value: str | None) -> str | None:
    """Treat blank strings as unset filter values."""
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None


def format_applied_filters(filters: SearchFilters) -> dict[str, str]:
    """Return only filters that are actively applied to the search."""
    applied: dict[str, str] = {}

    if filters.project:
        applied["project"] = filters.project
    if filters.lang:
        applied["lang"] = filters.lang
    if filters.path:
        applied["path"] = filters.path
    if filters.php_namespace:
        applied["php_namespace"] = filters.php_namespace
    if filters.php_class:
        applied["php_class"] = filters.php_class

    return applied


def build_qdrant_filter(filters: SearchFilters | None) -> models.Filter | None:
    if filters is None or filters.is_empty():
        return None

    must: list[models.Condition] = []

    if filters.project:
        must.append(
            models.FieldCondition(
                key="project",
                match=models.MatchValue(value=filters.project),
            )
        )

    if filters.lang:
        must.append(
            models.FieldCondition(
                key="language",
                match=models.MatchValue(value=filters.lang),
            )
        )

    if filters.path:
        must.append(
            models.FieldCondition(
                key="rel_path",
                match=models.MatchText(text=filters.path),
            )
        )

    if filters.php_namespace:
        must.append(
            models.FieldCondition(
                key="php_namespace",
                match=models.MatchValue(value=filters.php_namespace),
            )
        )

    if filters.php_class:
        must.append(
            models.FieldCondition(
                key="php_nearest_type_name",
                match=models.MatchValue(value=filters.php_class),
            )
        )

    if not must:
        return None

    return models.Filter(must=must)
