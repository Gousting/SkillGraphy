SkillGraph — Turn flat skill lists into a knowledge graph for AI agents.
Modern AI agents (Hermes, Claude Code, Cursor, Codex) use SKILL.md files to package specialized knowledge. As skill collections grow past hundreds, every skill gets injected into the system prompt every turn — wasting thousands of tokens. No one has built a retrieval layer for this.
SkillGraph constructs a knowledge graph from your skills using three edge types (explicit `related:`, same-category `sibling`, embedding-similarity `similar`), then retrieves only the relevant ones via embedding cosine matching + 1-hop graph traversal — with zero LLM calls and <50ms latency.
**147 skills → 8. ~4000 tokens → ~300 tokens. Per turn.**
Three embedding backends (local Ollama / offline sentence-transformers / OpenAI), framework-agnostic adapters, CLI + library + optional HTTP server. Works with any SKILL.md-based agent.
