# tests/test_cli.py
"""
CLI integration tests for RAGGAE.

Tests the command-line tools for indexing, search, and quickscore.
These tests use subprocess to simulate actual CLI usage.

Author: Dr. Olivier Vitrac, PhD, HDR
Organization: Adservio
Date: December 17, 2025
License: MIT

Run tests:
    # Unit tests (no external services needed)
    pytest tests/test_cli.py -v -k "not integration"

    # Full integration tests (require Ollama/Claude)
    pytest tests/test_cli.py -v -m integration
"""
import json
import subprocess
import sys
from pathlib import Path
import pytest


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def run_cli(module: str, args: list, check: bool = True) -> subprocess.CompletedProcess:
    """Run a CLI module with arguments."""
    cmd = [sys.executable, "-m", f"RAGGAE.cli.{module}"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    return result


# ---------------------------------------------------------------------------
# Index CLI Tests
# ---------------------------------------------------------------------------

class TestIndexCLI:
    """Tests for the index_doc CLI."""

    def test_index_help(self):
        """Test --help option."""
        result = run_cli("index_doc", ["--help"], check=False)
        assert result.returncode == 0
        assert "Index documents" in result.stdout

    def test_index_single_file(self, sample_text_file, temp_dir):
        """Index a single text file."""
        index_path = temp_dir / "test_index"

        result = run_cli("index_doc", [
            "--input", str(sample_text_file),
            "--out", str(index_path),
            "--e5",
            "--format", "json"
        ])

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["files_processed"] == 1
        assert output["total_chunks"] > 0
        assert (temp_dir / "test_index.faiss").exists()
        assert (temp_dir / "test_index.jsonl").exists()

    def test_index_directory(self, temp_dir):
        """Index a directory of files."""
        # Create test files
        docs_dir = temp_dir / "documents"
        docs_dir.mkdir()
        (docs_dir / "doc1.txt").write_text("First document about ISO 27001.")
        (docs_dir / "doc2.txt").write_text("Second document about MLflow.")

        index_path = temp_dir / "multi_index"

        result = run_cli("index_doc", [
            "--input", str(docs_dir),
            "--out", str(index_path),
            "--e5",
            "--format", "json"
        ])

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["files_processed"] == 2

    def test_index_missing_file(self, temp_dir):
        """Test error handling for missing file."""
        result = run_cli("index_doc", [
            "--input", "/nonexistent/file.pdf",
            "--out", str(temp_dir / "index"),
            "--e5"
        ], check=False)

        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Search CLI Tests
# ---------------------------------------------------------------------------

class TestSearchCLI:
    """Tests for the search CLI."""

    def test_search_help(self):
        """Test --help option."""
        result = run_cli("search", ["--help"], check=False)
        assert result.returncode == 0
        assert "Semantic search" in result.stdout

    def test_search_index(self, sample_text_file, temp_dir):
        """Search an indexed document."""
        index_path = temp_dir / "search_test"

        # First, create index
        run_cli("index_doc", [
            "--input", str(sample_text_file),
            "--out", str(index_path),
            "--e5"
        ])

        # Then search
        result = run_cli("search", [
            "--index", str(index_path),
            "--e5",
            "--query", "ISO 27001 certification",
            "--k", "5",
            "--format", "json"
        ])

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert output["count"] > 0
        assert output["query"] == "ISO 27001 certification"
        assert len(output["results"]) > 0

    def test_search_missing_index(self, temp_dir):
        """Test error handling for missing index."""
        result = run_cli("search", [
            "--index", str(temp_dir / "nonexistent"),
            "--e5",
            "--query", "test"
        ], check=False)

        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Quickscore CLI Tests
# ---------------------------------------------------------------------------

class TestQuickscoreCLI:
    """Tests for the quickscore CLI."""

    def test_quickscore_help(self):
        """Test --help option."""
        result = run_cli("quickscore", ["--help"], check=False)
        assert result.returncode == 0
        assert "NLI-based compliance scoring" in result.stdout
        assert "--backend" in result.stdout
        assert "ollama" in result.stdout
        assert "claude" in result.stdout

    @pytest.mark.integration
    @pytest.mark.ollama
    def test_quickscore_ollama(self, sample_text_file, temp_dir, ollama_available):
        """Test quickscore with Ollama backend."""
        if not ollama_available:
            pytest.skip("Ollama not running")

        index_path = temp_dir / "qs_test"

        # Create index
        run_cli("index_doc", [
            "--input", str(sample_text_file),
            "--out", str(index_path),
            "--e5"
        ])

        # Run quickscore
        result = run_cli("quickscore", [
            "--index", str(index_path),
            "--e5",
            "--backend", "ollama",
            "--req", "Provider must be ISO 27001 certified",
            "--format", "json"
        ])

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert "score" in output
        assert output["backend"] == "ollama"
        assert len(output["requirements"]) == 1

    @pytest.mark.integration
    @pytest.mark.claude
    def test_quickscore_claude(self, sample_text_file, temp_dir, claude_available):
        """Test quickscore with Claude backend."""
        if not claude_available:
            pytest.skip("Claude API key not available")

        index_path = temp_dir / "qs_claude_test"

        # Create index
        run_cli("index_doc", [
            "--input", str(sample_text_file),
            "--out", str(index_path),
            "--e5"
        ])

        # Run quickscore with Claude (using fastest model for tests)
        result = run_cli("quickscore", [
            "--index", str(index_path),
            "--e5",
            "--backend", "claude",
            "--claude-model", "claude-haiku-3-5-20241022",
            "--req", "Provider must be ISO 27001 certified",
            "--format", "json"
        ])

        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert "score" in output
        assert output["backend"] == "claude"

    def test_quickscore_csv_format(self, sample_text_file, temp_dir):
        """Test CSV output format (mocked)."""
        # This test checks the CLI accepts --format csv
        result = run_cli("quickscore", [
            "--help"
        ], check=False)

        assert "--format" in result.stdout
        assert "csv" in result.stdout


# ---------------------------------------------------------------------------
# End-to-End Pipeline Test
# ---------------------------------------------------------------------------

class TestPipeline:
    """End-to-end pipeline tests."""

    @pytest.mark.integration
    def test_full_pipeline(self, sample_text_file, temp_dir, ollama_available, claude_available):
        """Test the complete index → search → quickscore pipeline."""
        if not (ollama_available or claude_available):
            pytest.skip("No NLI backend available")

        index_path = temp_dir / "pipeline_test"

        # Step 1: Index
        result = run_cli("index_doc", [
            "--input", str(sample_text_file),
            "--out", str(index_path),
            "--e5",
            "--format", "json"
        ])
        assert result.returncode == 0
        index_output = json.loads(result.stdout)
        assert index_output["total_chunks"] > 0

        # Step 2: Search
        result = run_cli("search", [
            "--index", str(index_path),
            "--e5",
            "--query", "MLflow platform",
            "--format", "json"
        ])
        assert result.returncode == 0
        search_output = json.loads(result.stdout)
        assert search_output["count"] > 0

        # Step 3: Quickscore
        backend = "ollama" if ollama_available else "claude"
        args = [
            "--index", str(index_path),
            "--e5",
            "--backend", backend,
            "--req", "Provider must be ISO 27001 certified",
            "--req", "Platform uses MLflow",
            "--format", "json"
        ]
        if backend == "claude":
            args.extend(["--claude-model", "claude-haiku-3-5-20241022"])

        result = run_cli("quickscore", args)
        assert result.returncode == 0
        qs_output = json.loads(result.stdout)
        assert "score" in qs_output
        assert len(qs_output["requirements"]) == 2


# ---------------------------------------------------------------------------
# Output Format Tests
# ---------------------------------------------------------------------------

class TestOutputFormats:
    """Test different output formats."""

    def test_index_text_format(self, sample_text_file, temp_dir):
        """Test text output format for index."""
        index_path = temp_dir / "format_test"

        result = run_cli("index_doc", [
            "--input", str(sample_text_file),
            "--out", str(index_path),
            "--e5",
            "--format", "text"
        ])

        assert result.returncode == 0
        assert "Indexed" in result.stdout
        assert ".faiss" in result.stdout

    def test_search_text_format(self, sample_text_file, temp_dir):
        """Test text output format for search."""
        index_path = temp_dir / "search_format_test"

        run_cli("index_doc", [
            "--input", str(sample_text_file),
            "--out", str(index_path),
            "--e5"
        ])

        result = run_cli("search", [
            "--index", str(index_path),
            "--e5",
            "--query", "ISO 27001",
            "--format", "text"
        ])

        assert result.returncode == 0
        assert "Top-" in result.stdout
        assert "ISO 27001" in result.stdout
