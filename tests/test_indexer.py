"""Tests for the SKILL.md indexer."""

from pathlib import Path

from skillgraph.indexer import (
    SkillEntry,
    index_skills,
    parse_frontmatter,
    parse_category_descriptions,
    iter_skill_files,
)


class TestParseFrontmatter:
    def test_basic_frontmatter(self):
        content = """---
name: my-skill
description: A test skill
---

# My Skill

Body text.
"""
        meta, body = parse_frontmatter(content)
        assert meta["name"] == "my-skill"
        assert meta["description"] == "A test skill"
        assert "# My Skill" in body

    def test_no_frontmatter(self):
        content = "# Just markdown\n\nNo frontmatter here."
        meta, body = parse_frontmatter(content)
        assert meta == {}
        assert content in body

    def test_frontmatter_with_list(self):
        content = """---
name: excalidraw
related:
  - ascii-art
  - sketch
---

Body.
"""
        meta, _ = parse_frontmatter(content)
        assert meta["related"] == ["ascii-art", "sketch"]

    def test_empty_frontmatter(self):
        content = """---
---

Body.
"""
        meta, body = parse_frontmatter(content)
        assert meta == {}
        assert "Body." in body


class TestIndexSkills:
    def test_index_returns_entries(self, tmp_skills_dir: Path):
        entries = index_skills(tmp_skills_dir)
        assert len(entries) == 9  # Match SKILL_FIXTURES count

    def test_entry_has_name(self, tmp_skills_dir: Path):
        entries = index_skills(tmp_skills_dir)
        names = {e.name for e in entries}
        assert "excalidraw" in names
        assert "arxiv" in names
        assert "codegraph" in names

    def test_entry_has_category(self, tmp_skills_dir: Path):
        entries = index_skills(tmp_skills_dir)
        by_name = {e.name: e for e in entries}
        assert by_name["excalidraw"].category == "creative"
        assert by_name["arxiv"].category == "research"
        assert by_name["codegraph"].category == "software-development"

    def test_entry_has_related(self, tmp_skills_dir: Path):
        entries = index_skills(tmp_skills_dir)
        by_name = {e.name: e for e in entries}
        assert "ascii-art" in by_name["excalidraw"].related
        assert "sketch" in by_name["excalidraw"].related

    def test_entry_has_description(self, tmp_skills_dir: Path):
        entries = index_skills(tmp_skills_dir)
        by_name = {e.name: e for e in entries}
        assert "Hand-drawn" in by_name["excalidraw"].description

    def test_nonexistent_dir_returns_empty(self):
        entries = index_skills("/nonexistent/path")
        assert entries == []

    def test_deduplication(self, tmp_skills_dir: Path, tmp_path: Path):
        # Create a second dir with a duplicate skill
        ext_dir = tmp_path / "external"
        ext_dir.mkdir()
        (ext_dir / "excalidraw").mkdir()
        (ext_dir / "excalidraw" / "SKILL.md").write_text(
            "---\nname: excalidraw\ndescription: Different description\n---\nBody\n",
            encoding="utf-8",
        )
        entries = index_skills(tmp_skills_dir, external_dirs=[ext_dir])
        names = [e.name for e in entries]
        assert names.count("excalidraw") == 1  # Deduplicated

    def test_external_dir_skills_are_added(self, tmp_skills_dir: Path, tmp_path: Path):
        ext_dir = tmp_path / "external"
        (ext_dir / "new-skill").mkdir(parents=True)
        (ext_dir / "new-skill" / "SKILL.md").write_text(
            "---\nname: new-skill\ndescription: A new skill\n---\nBody\n",
            encoding="utf-8",
        )
        entries = index_skills(tmp_skills_dir, external_dirs=[ext_dir])
        names = {e.name for e in entries}
        assert "new-skill" in names
        # Original skills still present
        assert "excalidraw" in names


class TestSkillEntry:
    def test_to_index_dict(self):
        entry = SkillEntry(name="test", description="d", category="cat")
        d = entry.to_index_dict()
        assert d["name"] == "test"
        assert d["description"] == "d"
        assert d["category"] == "cat"
        assert d["related"] == []
        assert d["tools"] == []
        # Runtime fields excluded
        assert "score" not in d
        assert "embedding" not in d


class TestIterSkillFiles:
    def test_finds_all_skill_files(self, tmp_skills_dir: Path):
        files = iter_skill_files(tmp_skills_dir)
        assert len(files) == 9

    def test_nonexistent_dir(self):
        files = iter_skill_files(Path("/nonexistent"))
        assert files == []