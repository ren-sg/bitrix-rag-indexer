from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from bitrix_rag_indexer.app import index_source, search_query, show_stats
from bitrix_rag_indexer.search.filters import SearchFilters
from bitrix_rag_indexer.search.format_results import format_search_result
from bitrix_rag_indexer.eval.runner import run_eval
from bitrix_rag_indexer.app import index_source, prune_source, search_query, show_stats

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
    mode: str = typer.Option(
        None,
        "--mode",
        help="Search mode: dense, lexical, hybrid, qdrant-hybrid",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show rank/score internals for dense/lexical/hybrid results",
    ),
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
        mode=mode,
    )

    for item in results:
        console.print(format_search_result(item, debug=debug))


@app.command()
def stats(
    config_dir: Path = typer.Option(Path("configs"), help="Config directory"),
) -> None:
    """Show collection stats."""
    console.print(show_stats(config_dir=config_dir))

@app.command("eval")
def eval_command(
    profile: str = typer.Option("mvp", help="Config profile name"),
    config_dir: Path = typer.Option(Path("configs"), help="Config directory"),
    eval_file: Optional[Path] = typer.Option(
        None,
        "--file",
        help="Eval queries yaml file",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        help="Default search limit for eval cases",
    ),
    mode: str = typer.Option(
        None,
        "--mode",
        help="Search mode: dense, lexical, hybrid, qdrant-hybrid",
    ),
) -> None:
    """Evaluate search quality against expected paths."""
    result = run_eval(
        profile=profile,
        config_dir=config_dir,
        eval_file=eval_file,
        default_limit=limit,
        mode=mode,
    )

    console.print(
        f"[bold]Eval file:[/bold] {result['eval_file']}"
    )

    if result["total"] == 0:
        console.print("[yellow]No eval queries found.[/yellow]")
        return

    table = Table(title="Search eval")

    table.add_column("id")
    table.add_column("group")
    table.add_column("rank", justify="right")
    table.add_column("hit@5", justify="center")
    table.add_column("hit@10", justify="center")
    table.add_column("matched path")
    table.add_column("top paths")

    for case in result["cases"]:
        rank = case["first_rank"]
        rank_text = str(rank) if rank is not None else "-"

        table.add_row(
            case["id"],
            case["group"],
            rank_text,
            "yes" if case["hit_at_5"] else "no",
            "yes" if case["hit_at_10"] else "no",
            case["matched_path"] or "-",
            "\n".join(case["top_paths"]),
        )

    console.print(table)

    console.print(
        "[bold]Summary:[/bold] "
        f"total={result['total']}, "
        f"hit@5={result['hit_at_5']}/{result['total']} "
        f"({result['hit_at_5_rate']:.0%}), "
        f"hit@10={result['hit_at_10']}/{result['total']} "
        f"({result['hit_at_10_rate']:.0%})"
    )

    if result.get("by_group"):
        group_table = Table(title="By group")
        group_table.add_column("group")
        group_table.add_column("total", justify="right")
        group_table.add_column("hit@5", justify="right")
        group_table.add_column("hit@10", justify="right")

        for group, item in result["by_group"].items():
            total = item["total"]
            group_table.add_row(
                group,
                str(total),
                f"{item['hit_at_5']}/{total} ({item['hit_at_5_rate']:.0%})",
                f"{item['hit_at_10']}/{total} ({item['hit_at_10_rate']:.0%})",
            )

        console.print(group_table)

    failed_cases = result.get("failed_cases") or []
    if failed_cases:
        failed_table = Table(title="Failed cases")
        failed_table.add_column("id")
        failed_table.add_column("group")
        failed_table.add_column("query")
        failed_table.add_column("top paths")

        for case in failed_cases:
            failed_table.add_row(
                case["id"],
                case["group"],
                case["query"],
                "\n".join(case["top_paths"]),
            )

        console.print(failed_table)

@app.command()
def prune(
    profile: str = typer.Option("mvp", help="Config profile name"),
    source: str = typer.Option(..., "--source", help="Source name to prune"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show stale files without deleting"),
    config_dir: Path = typer.Option(Path("configs"), help="Config directory"),
) -> None:
    """Remove indexed files that no longer match source scan/exclude rules."""
    result = prune_source(
        profile=profile,
        source_name=source,
        config_dir=config_dir,
        dry_run=dry_run,
    )

    console.print(result)
