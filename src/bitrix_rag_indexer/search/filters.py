from dataclasses import dataclass

from qdrant_client import models


@dataclass(frozen=True)
class SearchFilters:
    source: str | None = None
    lang: str | None = None
    path: str | None = None
    source_type: str | None = None

    def is_empty(self) -> bool:
        return not any(
            [
                self.source,
                self.lang,
                self.path,
                self.source_type,
            ]
        )


def build_qdrant_filter(filters: SearchFilters | None) -> models.Filter | None:
    if filters is None or filters.is_empty():
        return None

    must: list[models.Condition] = []

    if filters.source:
        must.append(
            models.FieldCondition(
                key="source_name",
                match=models.MatchValue(value=filters.source),
            )
        )

    if filters.lang:
        must.append(
            models.FieldCondition(
                key="language",
                match=models.MatchValue(value=filters.lang),
            )
        )

    if filters.source_type:
        must.append(
            models.FieldCondition(
                key="source_type",
                match=models.MatchValue(value=filters.source_type),
            )
        )

    if filters.path:
        must.append(
            models.FieldCondition(
                key="rel_path",
                match=models.MatchText(text=filters.path),
            )
        )

    if not must:
        return None

    return models.Filter(must=must)
