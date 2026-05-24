from bitrix_rag_indexer.search.filters import (
    SearchFilters,
    format_applied_filters,
    normalize_search_lang,
    optional_filter_value,
)


def test_normalize_search_lang_aliases() -> None:
    assert normalize_search_lang("js") == "javascript"
    assert normalize_search_lang("JS") == "javascript"
    assert normalize_search_lang(" jsx ") == "javascript"
    assert normalize_search_lang("ts") == "typescript"
    assert normalize_search_lang("tsx") == "typescript"
    assert normalize_search_lang("md") == "markdown"
    assert normalize_search_lang("yml") == "yaml"


def test_search_filters_normalizes_lang_on_creation() -> None:
    filters = SearchFilters(project="my_project", lang="js")

    assert filters.lang == "javascript"


def test_normalize_search_lang_keeps_unknown_languages() -> None:
    assert normalize_search_lang("php") == "php"
    assert normalize_search_lang("vue") == "vue"
    assert normalize_search_lang(None) is None
    assert normalize_search_lang("   ") is None


def test_optional_filter_value_treats_blank_as_unset() -> None:
    assert optional_filter_value(None) is None
    assert optional_filter_value("") is None
    assert optional_filter_value("   ") is None
    assert optional_filter_value("bitrix_modules") == "bitrix_modules"


def test_format_applied_filters_omits_unset_values() -> None:
    filters = SearchFilters(lang="php", path="disk")

    assert format_applied_filters(filters) == {
        "lang": "php",
        "path": "disk",
    }


def test_format_applied_filters_empty() -> None:
    assert format_applied_filters(SearchFilters()) == {}
