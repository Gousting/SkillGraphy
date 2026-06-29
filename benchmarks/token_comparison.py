"""
Benchmark: compare system prompt token usage with and without SkillGraph.

Measures:
  1. Token count when all skills are injected (current Hermes behavior)
  2. Token count when only top-K skills are injected (SkillGraph)
  3. Retrieval latency
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from skillgraph import Retriever, create_embedder


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def format_all_skills(entries) -> str:
    """Format all skills as Hermes does (full list)."""
    lines = ["<available_skills>"]
    by_cat: dict[str, list] = {}
    for e in entries:
        by_cat.setdefault(e.category, []).append(e)
    for cat in sorted(by_cat):
        lines.append(f"  {cat}:")
        for e in sorted(by_cat[cat], key=lambda x: x.name):
            if e.description:
                lines.append(f"    - {e.name}: {e.description}")
            else:
                lines.append(f"    - {e.name}")
    lines.append("</available_skills>")
    return "\n".join(lines)


def format_top_k(skills) -> str:
    """Format only top-K skills."""
    lines = ["<available_skills>"]
    by_cat: dict[str, list] = {}
    for e in skills:
        by_cat.setdefault(e.category, []).append(e)
    for cat in sorted(by_cat):
        lines.append(f"  {cat}:")
        for e in sorted(by_cat[cat], key=lambda x: -x.score):
            if e.description:
                lines.append(f"    - {e.name}: {e.description}")
            else:
                lines.append(f"    - {e.name}")
    lines.append("</available_skills>")
    return "\n".join(lines)


def main():
    skills_dir = os.path.expanduser("~/.hermes/skills")
    if not Path(skills_dir).exists():
        print(f"Skills dir not found: {skills_dir}")
        print("Point SKILLGRAPH_SKILLS_DIR to your skills directory.")
        sys.exit(1)

    embedder = create_embedder("ollama")
    retriever = Retriever(skills_dir=skills_dir, embedder=embedder)
    retriever.build()

    queries = [
        "generate an architecture diagram",
        "search for arXiv papers about transformers",
        "deploy a minecraft modded server",
        "write a blog post about AI music",
        "set up a webhook subscription for my agent",
        "debug a Python async issue",
        "create pixel art for a game",
        "monitor RSS feeds for AI news",
    ]

    all_tokens = estimate_tokens(format_all_skills(retriever.entries))
    print(f"\n{'Query':50s} | All Skills | Top-8 | Savings | Latency")
    print(f"{'-'*50}-+-{'-'*11}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")

    latencies = []
    for query in queries:
        t0 = time.perf_counter()
        results = retriever.retrieve(query, top_k=8)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        top_prompt = format_top_k(results)
        top_tokens = estimate_tokens(top_prompt)
        savings = all_tokens - top_tokens

        print(
            f"{query:50s} | {all_tokens:9d} | {top_tokens:5d} | {savings:6d} | {elapsed:6.1f}ms"
        )

    avg_lat = sum(latencies) / len(latencies)
    print(f"\nAverage retrieval latency: {avg_lat:.1f}ms")
    print(f"Token savings per query: ~{all_tokens - all_tokens // len(retriever.entries) * 8}")


if __name__ == "__main__":
    main()