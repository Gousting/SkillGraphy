"""
Basic usage example for SkillGraph.

Demonstrates:
  1. Building an index from a skills directory
  2. Querying for relevant skills
  3. Using graph expansion
  4. Saving/loading the index
"""

from pathlib import Path
import os
import sys

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from skillgraph import Retriever, create_embedder


def main():
    # 1. Create an embedder (ollama is default — needs Ollama running)
    #    Alternatives: "local" (sentence-transformers, offline) or "openai"
    print("Creating embedder...")
    embedder = create_embedder("ollama")

    # 2. Point to your skills directory
    skills_dir = os.path.expanduser("~/.hermes/skills")
    if not Path(skills_dir).exists():
        print(f"Skills dir not found: {skills_dir}")
        print("Creating a small demo directory...")
        skills_dir = str(Path(__file__).parent / "demo_skills")
        _create_demo_skills(skills_dir)

    # 3. Create retriever and build index
    print(f"Building index from {skills_dir}...")
    retriever = Retriever(skills_dir=skills_dir, embedder=embedder)
    count = retriever.build()
    print(f"  → Indexed {count} skills")

    # 4. Print stats
    stats = retriever.stats
    print(f"\nIndex stats:")
    print(f"  Embedder: {stats['embedder']}")
    print(f"  Dimensions: {stats['embedding_dim']}")
    print(f"  Categories: {stats['categories']}")
    print(f"  Graph: {stats['graph']['nodes']} nodes, {stats['graph']['edges']} edges")
    print(f"  Edge types: {stats['graph']['edge_types']}")

    # 5. Query for skills
    queries = [
        "generate a hand-drawn diagram of my architecture",
        "search for academic papers about language models",
        "set up a minecraft server for my friends",
        "create ASCII art from an image",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        results = retriever.retrieve(query, top_k=5)
        for i, skill in enumerate(results, 1):
            print(f"  {i}. {skill.name:30s} score={skill.score:.3f} [{skill.source}]")
            if skill.description:
                print(f"     {skill.description[:80]}")


def _create_demo_skills(skills_dir: str):
    """Create a small set of demo skills for testing."""
    import yaml
    base = Path(skills_dir)
    base.mkdir(parents=True, exist_ok=True)

    demo_skills = {
        "creative/excalidraw": {
            "name": "excalidraw",
            "description": "Hand-drawn Excalidraw JSON diagrams (arch, flow, seq).",
            "related": ["ascii-art", "sketch"],
        },
        "creative/ascii-art": {
            "name": "ascii-art",
            "description": "ASCII art: pyfiglet, cowsay, boxes, image-to-ascii.",
            "related": ["excalidraw"],
        },
        "gaming/minecraft": {
            "name": "minecraft-modpack-server",
            "description": "Host modded Minecraft servers (CurseForge, Modrinth).",
        },
        "research/arxiv": {
            "name": "arxiv",
            "description": "Search arXiv papers by keyword, author, category, or ID.",
        },
    }

    for rel_path, meta in demo_skills.items():
        path = base / rel_path / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        fm = yaml.dump(meta, default_flow_style=False)
        path.write_text(f"---\n{fm}---\n\n# {meta['name']}\n", encoding="utf-8")


if __name__ == "__main__":
    main()