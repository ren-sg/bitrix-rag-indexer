from bitrix_rag_indexer.search.filters import SearchFilters, normalize_search_lang


def test_normalize_search_lang_aliases() -> None:
    assert normalize_search_lang("js") == "javascript"
    assert normalize_search_lang("JS") == "javascript"
    assert normalize_search_lang(" jsx ") == "javascript"
    assert normalize_search_lang("ts") == "typescript"
    assert normalize_search_lang("tsx") == "typescript"
    assert normalize_search_lang("md") == "markdown"
    assert normalize_search_lang("yml") == "yaml"


def test_search_filters_normalizes_lang_on_creation() -> None:
    filters = SearchFilters(source="project_local", lang="js")

    assert filters.lang == "javascript"


def test_normalize_search_lang_keeps_unknown_languages() -> None:
    assert normalize_search_lang("php") == "php"
    assert normalize_search_lang("vue") == "vue"
    assert normalize_search_lang(None) is None
    assert normalize_search_lang("   ") is None
