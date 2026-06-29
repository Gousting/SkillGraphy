<p align="center">
  <h1 align="center">SkillGraph</h1>
  <p align="center">Turn flat skill lists into a knowledge graph. Dynamic skill retrieval for AI agents.</p>
</p>

<p align="center">
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-beta-orange.svg">
</p>

---

## Why?

Modern AI agents (Hermes, Claude Code, Cursor, Codex) use **SKILL.md** files to package
specialized knowledge. But as skill collections grow past dozens or hundreds:

- **Every skill** gets injected into the system prompt every turn — wasting thousands of tokens
- The agent must **scan all skills** to decide which to load, adding latency and noise
- Related skills (e.g. `ascii-art`, `excalidraw`, `sketch`) have **no explicit connections**
- Skill marketplaces (SkillsMP 175k+ skills) have **no retrieval layer**

**SkillGraph solves this** by building a knowledge graph from your skills and retrieving
only the relevant ones — using embedding similarity + graph traversal, with zero LLM calls.

## How It Works

```
User message
    │
    ▼  embed (local Ollama / OpenAI / sentence-transformers)
┌──────────────┐
│ Skill Vector  │  ← pre-computed embeddings for all SKILL.md files
│ Index (N)     │
└──────┬───────┘
       │ top-3 cosine similarity
       ▼
┌──────────────┐
│ Skill Graph   │  ← edges: related / sibling / similar
│ (adjacency)   │     from frontmatter + category + embeddings
└──────┬───────┘
       │ 1-hop neighbor expansion
       ▼
┌──────────────┐
│ Rerank        │  alpha * semantic + beta * graph_weight
│ → top-K (8)   │
└──────────────┘
       │
       ▼  inject into system prompt
   Agent receives only relevant skills
```

## Features

- **147 skills → 8**: Reduce system prompt from ~4000 tokens to ~300 tokens
- **Zero LLM calls**: Pure vector math + graph traversal, latency < 50ms
- **Graph-enhanced recall**: Match `excalidraw` → automatically surface `sketch` + `ascii-art`
- **Local-first**: Ollama embedding backend needs no API key, runs offline
- **Framework-agnostic**: Works with any SKILL.md-based agent (Hermes, Claude Code, Cursor, Codex)
- **Three embedding backends**: Ollama · OpenAI · sentence-transformers (local)
- **Three edge types**: `related` (explicit frontmatter) · `sibling` (same category) · `similar` (cosine threshold)

## Quick Start

### Install

```bash
pip install skillgraph

# Optional backends
pip install "skillgraph[local]"    # sentence-transformers (offline)
pip install "skillgraph[openai]"   # OpenAI embeddings
```

### Build the index from your skills directory

```bash
# Using Ollama (default — needs Ollama running on localhost:11434)
skillgraph build --skills-dir ~/.hermes/skills --backend ollama

# Using sentence-transformers (fully offline)
skillgraph build --skills-dir ~/.hermes/skills --backend local

# Using OpenAI
skillgraph build --skills-dir ~/.hermes/skills --backend openai --api-key sk-...
```

### Query for relevant skills

```bash
skillgraph query "help me generate ASCII art"
skillgraph query "deploy a minecraft server" --top-k 5
skillgraph query "write a blog post about AI" --json
```

### Use as a library

```python
from skillgraph import SkillGraph

sg = SkillGraph(
    skills_dir="~/.hermes/skills",
    backend="ollama",
)
sg.build()

results = sg.retrieve("generate a hand-drawn diagram", top_k=5)
for skill in results:
    print(f"{skill.name}: {skill.description}  (score={skill.score:.3f})")
```

### Add `related` edges to your SKILL.md

```yaml
---
name: excalidraw
description: Hand-drawn Excalidraw JSON diagrams
related:
  - ascii-art
  - sketch
  - architecture-diagram
---
```

## CLI Reference

```bash
skillgraph build   --skills-dir DIR [--backend ollama|local|openai] [--api-key KEY]
skillgraph query   "natural language query" [--top-k N] [--json]
skillgraph info     # show index stats
skillgraph graph    # visualize graph (text output)
skillgraph rebuild  # force full rebuild
```

## Adapter Integration

SkillGraph provides adapters for common agent frameworks:

```python
from skillgraph.adapters import HermesAdapter, GenericAdapter

# Hermes Agent — replaces build_skills_system_prompt()
adapter = HermesAdapter(skills_dir="~/.hermes/skills", backend="ollama")
prompt = adapter.build_prompt(user_message="generate a diagram")
# → only 8 relevant skills in the system prompt

# Any SKILL.md setup
adapter = GenericAdapter(skills_dir="./skills", backend="local")
```

## Architecture

```
skillgraph/
├── indexer.py      # SKILL.md scanner + frontmatter parser
├── embedder.py     # Embedding backends (Ollama / OpenAI / local)
├── graph.py        # Knowledge graph builder (related / sibling / similar edges)
├── retriever.py     # Semantic matching + graph traversal retrieval
├── adapters/       # Framework integrations
│   ├── base.py
│   ├── hermes.py
│   ├── claude_code.py
│   └── generic.py
├── cli.py          # CLI entry point
└── server.py       # Optional HTTP server
```

## Edge Types

| Edge | Source | Default Weight | Example |
|------|--------|---------------|---------|
| `related` | SKILL.md frontmatter `related:` | 1.0 | `excalidraw → sketch` |
| `sibling` | Same `category` | 0.3 | `ascii-art ↔ excalidraw` |
| `similar` | Embedding cosine > threshold | cosine | `pixel-art → sketch` (0.78) |

## Benchmark

| Method | System Prompt Tokens | Retrieval Latency | LLM Calls |
|--------|--------------------|-------------------|-----------|
| Inject all (current) | ~4000 | 0ms | 0 |
| SkillGraph (top-8) | ~300 | <50ms | 0 |
| LLM-based selection | ~300 | 500-2000ms | 1 |

## Contributing

Contributions welcome! Areas of interest:

- More adapters (Cursor, Codex, Windsurf, custom frameworks)
- Better embedding models / reranking strategies
- Graph-based skill recommendations
- Visualization tools

## License

MIT — see [LICENSE](./LICENSE).