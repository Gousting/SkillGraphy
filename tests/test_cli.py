"""Tests for the CLI."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from skillgraph.cli import main


class TestCLI:
    def test_build_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["build", "--help"])
        assert result.exit_code == 0
        assert "--skills-dir" in result.output
        assert "--backend" in result.output

    def test_query_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["query", "--help"])
        assert result.exit_code == 0
        assert "--top-k" in result.output
        assert "--json" in result.output

    def test_info_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["info", "--help"])
        assert result.exit_code == 0

    def test_build_no_skills_dir_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(main, ["build", "--skills-dir", "/nonexistent/path", "--backend", "ollama"])
        assert result.exit_code != 0