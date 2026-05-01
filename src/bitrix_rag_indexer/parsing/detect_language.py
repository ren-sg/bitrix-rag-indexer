from pathlib import Path


EXTENSION_TO_LANGUAGE = {
    ".php": "php",
    ".js": "javascript",
    ".ts": "typescript",
    ".vue": "vue",
    ".html": "html",
    ".htm": "html",
    ".md": "markdown",
}


def detect_language(path: Path) -> str:
    return EXTENSION_TO_LANGUAGE.get(path.suffix.lower(), "text")
