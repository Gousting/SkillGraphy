"""Adapters for integrating SkillGraph with various agent frameworks."""

from .base import BaseAdapter
from .hermes import HermesAdapter
from .generic import GenericAdapter
from .claude_code import ClaudeCodeAdapter

__all__ = [
    "BaseAdapter",
    "HermesAdapter",
    "GenericAdapter",
    "ClaudeCodeAdapter",
]