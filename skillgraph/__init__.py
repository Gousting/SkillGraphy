"""
SkillGraph — Turn flat skill lists into a knowledge graph.

Dynamic skill retrieval for AI agents using embedding similarity + graph traversal.
"""

from .indexer import SkillEntry, index_skills, parse_frontmatter
from .embedder import Embedder, create_embedder
from .graph import SkillGraph, Edge
from .retriever import Retriever

__version__ = "0.1.0"
__all__ = [
    "SkillEntry",
    "SkillGraph",
    "Edge",
    "Retriever",
    "Embedder",
    "create_embedder",
    "index_skills",
    "parse_frontmatter",
]