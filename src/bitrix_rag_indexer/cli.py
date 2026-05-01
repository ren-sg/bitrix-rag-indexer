from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from bitrix_rag_indexer.app import index_source, search_query, show_stats

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def index(
    profile: str = typer.Option("mvp", help="Config profile name"),
    source: Optional[str] = typer.Option(None, help="Index only selected source"),
    force: bool = typer.Option(False, "--force", help="Reindex unchanged files"),
    config_dir: Path = typer.Option(Path("configs"), help="Config directory"),
) -> None:
    """Index configured sources."""
    result = index_source(
        profile=profile,
        source_name=source,
        force=force,
        config_dir=config_dir,
    )
    console.print(result)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, help="Number of results"),
    config_dir: Path = typer.Option(Path("configs"), help="Config directory"),
) -> None:
    """Search indexed chunks."""
    results = search_query(query=query, limit=limit, config_dir=config_dir)
    for item in results:
        console.rule(f"[bold]{item['score']:.4f}[/bold] {item['path']}")
        console.print(item["text"][:1200])


@app.command()
def stats(
    config_dir: Path = typer.Option(Path("configs"), help="Config directory"),
) -> None:
    """Show collection stats."""
    console.print(show_stats(config_dir=config_dir))
