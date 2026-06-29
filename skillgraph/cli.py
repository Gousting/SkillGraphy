"""
SkillGraph CLI — build, query, and inspect your skill knowledge graph.

Usage:
  skillgraph build   --skills-dir DIR [--backend ollama|local|openai]
  skillgraph query   "natural language" [--top-k N] [--json]
  skillgraph info     # show index stats
  skillgraph graph    # visualize graph (text)
  skillgraph rebuild  # force full rebuild
"""

from __future__ import annotations

import json as json_lib
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# Default storage location
DEFAULT_STORAGE = Path(os.environ.get(
    "SKILLGRAPH_HOME",
    os.path.join(os.path.expanduser("~"), ".skillgraph"),
))


def _get_retriever(
    skills_dir: str,
    backend: str,
    api_key: str | None = None,
    storage_dir: Path | None = None,
) -> "Retriever":
    """Create and load/build a retriever."""
    from .retriever import Retriever
    from .embedder import create_embedder

    storage = storage_dir or (DEFAULT_STORAGE / Path(skills_dir).resolve().name)
    storage.mkdir(parents=True, exist_ok=True)

    embedder = create_embedder(backend, api_key=api_key)

    retriever = Retriever(
        skills_dir=skills_dir,
        embedder=embedder,
    )

    # Try loading existing index
    if (storage / "skill_index.json").exists() and (storage / "embeddings.npy").exists():
        retriever.load(storage)
        return retriever

    # Build fresh
    count = retriever.build()
    if count == 0:
        console.print(f"[red]No skills found in {skills_dir}[/red]")
        sys.exit(1)
    retriever.save(storage)
    return retriever


# ── CLI Group ───────────────────────────────────────────────────────────


@click.group()
def main() -> None:
    """SkillGraph — Dynamic skill retrieval for AI agents."""
    pass


# ── build ──────────────────────────────────────────────────────────────


@main.command()
@click.option("--skills-dir", required=True, help="Path to skills directory")
@click.option("--backend", default="ollama", type=click.Choice(["ollama", "local", "openai"]))
@click.option("--api-key", default=None, help="API key (openai only)")
@click.option("--external-dir", multiple=True, help="Additional skill dirs (can repeat)")
def build(
    skills_dir: str,
    backend: str,
    api_key: str | None,
    external_dir: tuple[str, ...],
) -> None:
    """Build the skill index and knowledge graph."""
    from .embedder import create_embedder
    from .retriever import Retriever

    embedder = create_embedder(backend, api_key=api_key)

    retriever = Retriever(
        skills_dir=skills_dir,
        external_dirs=list(external_dir) if external_dir else None,
        embedder=embedder,
    )

    with console.status(f"[bold green]Building index from {skills_dir}..."):
        count = retriever.build()

    if count == 0:
        console.print(f"[red]No skills found in {skills_dir}[/red]")
        sys.exit(1)

    # Save index
    storage = DEFAULT_STORAGE / Path(skills_dir).resolve().name
    retriever.save(storage)

    stats = retriever.stats
    console.print(Panel.fit(
        f"[bold green]✅ Built index: {count} skills[/bold green]\n"
        f"  Embedder: {stats['embedder']}\n"
        f"  Dimensions: {stats['embedding_dim']}\n"
        f"  Categories: {len(stats['categories'])}\n"
        f"  Graph nodes: {stats['graph']['nodes']}\n"
        f"  Graph edges: {stats['graph']['edges']}\n"
        f"  Saved to: {storage}",
        title="SkillGraph Build",
    ))


# ── query ──────────────────────────────────────────────────────────────


@main.command()
@click.argument("query", nargs=-1, required=True)
@click.option("--skills-dir", default=None, help="Path to skills directory")
@click.option("--backend", default="ollama", type=click.Choice(["ollama", "local", "openai"]))
@click.option("--api-key", default=None)
@click.option("--top-k", default=8, help="Number of skills to retrieve")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--no-graph", is_flag=True, help="Disable graph expansion")
def query(
    query: tuple[str, ...],
    skills_dir: str | None,
    backend: str,
    api_key: str | None,
    top_k: int,
    as_json: bool,
    no_graph: bool,
) -> None:
    """Query for relevant skills."""
    skills_dir = skills_dir or os.path.join(
        os.path.expanduser("~"), ".hermes", "skills"
    )
    query_text = " ".join(query)

    try:
        retriever = _get_retriever(skills_dir, backend, api_key)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print(
            f"[yellow]Run `skillgraph build --skills-dir {skills_dir} --backend {backend}` first.[/yellow]"
        )
        sys.exit(1)

    import time
    t0 = time.perf_counter()
    results = retriever.retrieve(query_text, top_k=top_k, expand_graph=not no_graph)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if as_json:
        output = [
            {
                "name": s.name,
                "description": s.description,
                "category": s.category,
                "score": round(s.score, 4),
                "source": s.source,
            }
            for s in results
        ]
        click.echo(json_lib.dumps(output, indent=2, ensure_ascii=False))
        return

    # Rich table output
    console.print()
    console.print(f"[bold]Query:[/bold] {query_text}")
    console.print(f"[dim]Retrieved {len(results)} skills in {elapsed_ms:.1f}ms[/dim]")
    console.print()

    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Skill", style="bold cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Source", style="dim")
    table.add_column("Description")

    for i, skill in enumerate(results, 1):
        table.add_row(
            str(i),
            skill.name,
            skill.category,
            f"{skill.score:.3f}",
            skill.source,
            skill.description[:60] + "..." if len(skill.description) > 60 else skill.description,
        )

    console.print(table)


# ── info ──────────────────────────────────────────────────────────────


@main.command()
@click.option("--skills-dir", default=None)
@click.option("--backend", default="ollama", type=click.Choice(["ollama", "local", "openai"]))
def info(skills_dir: str | None, backend: str) -> None:
    """Show index statistics."""
    skills_dir = skills_dir or os.path.join(
        os.path.expanduser("~"), ".hermes", "skills"
    )

    try:
        retriever = _get_retriever(skills_dir, backend)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    stats = retriever.stats

    # Overview panel
    console.print(Panel.fit(
        f"[bold]{stats['total_skills']}[/bold] skills indexed\n"
        f"  Embedder: [cyan]{stats['embedder']}[/cyan]\n"
        f"  Dimensions: {stats['embedding_dim']}\n"
        f"  Graph: {stats['graph']['nodes']} nodes, {stats['graph']['edges']} edges",
        title="SkillGraph Index",
    ))

    # Category breakdown
    cat_table = Table(title="Categories", box=box.SIMPLE)
    cat_table.add_column("Category", style="magenta")
    cat_table.add_column("Skills", justify="right", style="cyan")
    for cat, count in sorted(stats["categories"].items(), key=lambda x: -x[1]):
        cat_table.add_row(cat, str(count))
    console.print(cat_table)

    # Edge types
    edge_table = Table(title="Graph Edges", box=box.SIMPLE)
    edge_table.add_column("Type", style="yellow")
    edge_table.add_column("Count", justify="right")
    for etype, count in stats["graph"]["edge_types"].items():
        edge_table.add_row(etype, str(count))
    console.print(edge_table)


# ── graph visualize ───────────────────────────────────────────────────


@main.command(name="graph")
@click.option("--skills-dir", default=None)
@click.option("--backend", default="ollama", type=click.Choice(["ollama", "local", "openai"]))
@click.option("--skill", default=None, help="Show neighbors of a specific skill")
def graph_cmd(skills_dir: str | None, backend: str, skill: str | None) -> None:
    """Visualize the skill graph (text output)."""
    skills_dir = skills_dir or os.path.join(
        os.path.expanduser("~"), ".hermes", "skills"
    )

    try:
        retriever = _get_retriever(skills_dir, backend)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    if retriever.graph is None:
        console.print("[red]Graph not built.[/red]")
        sys.exit(1)

    if skill:
        # Show neighbors of a specific skill
        edges = retriever.graph.get_neighbors(skill)
        if not edges:
            console.print(f"[red]Skill '{skill}' not found or has no edges.[/red]")
            sys.exit(1)

        console.print(f"\n[bold cyan]{skill}[/bold cyan] → {len(edges)} neighbors:")
        for edge in sorted(edges, key=lambda e: -e.weight):
            color = {"related": "bold green", "sibling": "yellow", "similar": "blue"}.get(
                edge.edge_type, "white"
            )
            console.print(
                f"  [{color}]{edge.edge_type:8s}[/{color}] "
                f"→ [bold]{edge.target}[/bold] "
                f"(w={edge.weight:.3f})"
            )
    else:
        # Show top-connected nodes
        adj = retriever.graph.adjacency
        top_nodes = sorted(adj.items(), key=lambda x: len(x[1]), reverse=True)[:20]

        console.print(f"\n[bold]Top 20 most connected skills:[/bold]")
        for name, edges in top_nodes:
            console.print(
                f"  [bold cyan]{name:30s}[/bold cyan] "
                f"[dim]{len(edges)} edges[/dim]"
            )


# ── rebuild ─────────────────────────────────────────────────────────────


@main.command()
@click.option("--skills-dir", required=True)
@click.option("--backend", default="ollama", type=click.Choice(["ollama", "local", "openai"]))
@click.option("--api-key", default=None)
def rebuild(skills_dir: str, backend: str, api_key: str | None) -> None:
    """Force a full rebuild of the index."""
    storage = DEFAULT_STORAGE / Path(skills_dir).resolve().name
    # Clear existing index
    for f in ["skill_index.json", "embeddings.npy", "adjacency.json"]:
        (storage / f).unlink(missing_ok=True)
    # Rebuild
    ctx = click.get_current_context()
    ctx.invoke(build, skills_dir=skills_dir, backend=backend, api_key=api_key, external_dir=())


if __name__ == "__main__":
    main()