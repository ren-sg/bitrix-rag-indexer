from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from bitrix_rag_indexer.app import index_source, search_query, show_stats
from bitrix_rag_indexer.search.filters import SearchFilters
from bitrix_rag_indexer.search.format_results import format_search_result

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def index(
    profile: str = typer.Option("mvp", help="Config profile name"),
    source: Optional[str] = typer.Option(None, help="Index only selected source"),
    force: bool = typer.Option(False, "--force", help="Reindex unchanged files"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Scan files without indexing"),
    max_files: Optional[int] = typer.Option(
        None,
        "--max-files",
        help="Index only first N files from selected source",
    ),
    config_dir: Path = typer.Option(Path("configs"), help="Config directory"),
) -> None:

    """Index configured sources."""
    result = index_source(
        profile=profile,
        source_name=source,
        force=force,
        dry_run=dry_run,
        max_files=max_files,
        config_dir=config_dir,
    )
    console.print(result)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of results"),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by source_name"),
    lang: Optional[str] = typer.Option(None, "--lang", help="Filter by language"),
    path: Optional[str] = typer.Option(None, "--path", help="Filter by rel_path text"),
    score_threshold: Optional[float] = typer.Option(
        None,
        "--score-threshold",
        help="Minimal Qdrant score",
    ),
    config_dir: Path = typer.Option(Path("configs"), help="Config directory"),
) -> None:
    """Search indexed chunks."""
    filters = SearchFilters(
        source=source,
        lang=lang,
        path=path,
    )

    results = search_query(
        query=query,
        limit=limit,
        config_dir=config_dir,
        score_threshold=score_threshold,
        filters=filters,
    )

    for item in results:
        console.print(format_search_result(item))


@app.command()
def stats(
    config_dir: Path = typer.Option(Path("configs"), help="Config directory"),
) -> None:
    """Show collection stats."""
    console.print(show_stats(config_dir=config_dir))
