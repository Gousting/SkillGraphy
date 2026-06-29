"""Tests for framework adapters."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from skillgraph.adapters import HermesAdapter, GenericAdapter, ClaudeCodeAdapter
from skillgraph.indexer import SkillEntry


class TestHermesAdapter:
    def test_get_skills_dir(self):
        adapter = HermesAdapter()
        skills_dir = adapter.get_skills_dir()
        assert "skills" in str(skills_dir)

    def test_format_prompt_with_skills(self):
        adapter = HermesAdapter(skills_dir="/nonexistent")
        skills = [
            SkillEntry(name="ascii-art", description="ASCII art", category="creative", score=0.9),
            SkillEntry(name="kanban-orchestrator", description="Kanban", category="devops", score=0.7),
        ]
        prompt = adapter.format_prompt(skills)
        assert "<available_skills>" in prompt
        assert "ascii-art" in prompt
        assert "kanban-orchestrator" in prompt
        assert "ASCII art" in prompt

    def test_format_prompt_empty(self):
        adapter = HermesAdapter(skills_dir="/nonexistent")
        prompt = adapter.format_prompt([])
        assert prompt == ""

    def test_format_prompt_groups_by_category(self):
        adapter = HermesAdapter(skills_dir="/nonexistent")
        skills = [
            SkillEntry(name="a", description="d1", category="creative", score=0.9),
            SkillEntry(name="b", description="d2", category="devops", score=0.8),
            SkillEntry(name="c", description="d3", category="creative", score=0.7),
        ]
        prompt = adapter.format_prompt(skills)
        # creative should appear before devops (alphabetical)
        assert prompt.index("creative") < prompt.index("devops")


class TestGenericAdapter:
    def test_requires_skills_dir(self):
        adapter = GenericAdapter(skills_dir="/tmp/skills")
        assert str(adapter.get_skills_dir()) == "/tmp/skills"

    def test_format_prompt(self):
        adapter = GenericAdapter(skills_dir="/nonexistent")
        skills = [
            SkillEntry(name="test", description="A test skill", category="cat", score=0.85),
        ]
        prompt = adapter.format_prompt(skills)
        assert "test" in prompt
        assert "A test skill" in prompt
        assert "0.850" in prompt or "0.85" in prompt


class TestClaudeCodeAdapter:
    def test_get_skills_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        adapter = ClaudeCodeAdapter()
        skills_dir = adapter.get_skills_dir()
        assert ".claude" in str(skills_dir)
        assert "skills" in str(skills_dir)

    def test_format_prompt(self):
        adapter = ClaudeCodeAdapter(skills_dir="/nonexistent")
        skills = [
            SkillEntry(name="test", description="A test", category="cat", score=0.9),
        ]
        prompt = adapter.format_prompt(skills)
        assert "test" in prompt
        assert "A test" in prompt
        assert "Relevant Skills" in prompt