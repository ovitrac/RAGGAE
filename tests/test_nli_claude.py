# tests/test_nli_claude.py
"""
Tests for Claude NLI client.

Includes:
- Unit tests with mocked responses (no API key needed)
- Integration tests (require ANTHROPIC_API_KEY)
- Config file loading tests

Author: Dr. Olivier Vitrac, PhD, HDR
Organization: Adservio
Date: December 17, 2025
License: MIT

Run tests:
    # Unit tests only (no API key needed)
    pytest tests/test_nli_claude.py -v -k "not integration"

    # All tests (requires ANTHROPIC_API_KEY)
    pytest tests/test_nli_claude.py -v
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from RAGGAE.core.nli_claude import (
    ClaudeNLIClient,
    ClaudeNLIConfig,
    load_api_key,
    save_api_key,
    get_config_path,
    create_nli_client,
    _parse_json_response,
    CONFIG_PATHS,
)
from RAGGAE.core.nli_ollama import NLIResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_anthropic():
    """Mock the anthropic SDK."""
    with patch("RAGGAE.core.nli_claude._get_anthropic") as mock:
        mock_client = MagicMock()
        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        mock.return_value = mock_anthropic_module
        yield mock_client


@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ---------------------------------------------------------------------------
# Unit Tests: JSON Parsing
# ---------------------------------------------------------------------------

class TestJsonParsing:
    """Test JSON response parsing from Claude."""

    def test_parse_simple_json(self):
        """Parse plain JSON response."""
        text = '{"label": "Yes", "rationale": "The clause matches"}'
        result = _parse_json_response(text)
        assert result == {"label": "Yes", "rationale": "The clause matches"}

    def test_parse_json_with_markdown(self):
        """Parse JSON wrapped in markdown code block."""
        text = '```json\n{"label": "Partial", "rationale": "Partially matches"}\n```'
        result = _parse_json_response(text)
        assert result == {"label": "Partial", "rationale": "Partially matches"}

    def test_parse_json_array(self):
        """Parse JSON array response (batch mode)."""
        text = '[{"label": "Yes", "rationale": "Match 1"}, {"label": "No", "rationale": "No match"}]'
        result = _parse_json_response(text)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["label"] == "Yes"

    def test_parse_invalid_json(self):
        """Return None for invalid JSON."""
        text = "This is not JSON at all"
        result = _parse_json_response(text)
        assert result is None

    def test_parse_json_with_surrounding_text(self):
        """Parse JSON even with surrounding text."""
        text = 'Here is the result:\n{"label": "Yes", "rationale": "OK"}\nDone.'
        result = _parse_json_response(text)
        assert result["label"] == "Yes"


# ---------------------------------------------------------------------------
# Unit Tests: Config File Loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    """Test API key loading from various sources."""

    def test_explicit_key_priority(self):
        """Explicit key takes priority over env/config."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            key = load_api_key("explicit-key")
            assert key == "explicit-key"

    def test_env_var_fallback(self):
        """Environment variable used when no explicit key."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}, clear=False):
            # Clear any explicit key
            key = load_api_key(None)
            assert key == "env-key"

    def test_config_file_fallback(self, temp_config_dir):
        """Config file used when no explicit key or env var."""
        config_file = temp_config_dir / "config.json"
        config_file.write_text('{"anthropic_api_key": "config-key"}')

        # Patch CONFIG_PATHS to use our temp dir
        with patch("RAGGAE.core.nli_claude.CONFIG_PATHS", [config_file]):
            with patch.dict(os.environ, {}, clear=True):
                # Remove ANTHROPIC_API_KEY from env
                os.environ.pop("ANTHROPIC_API_KEY", None)
                key = load_api_key(None)
                assert key == "config-key"

    def test_no_key_found(self, temp_config_dir):
        """Return None when no key found anywhere."""
        with patch("RAGGAE.core.nli_claude.CONFIG_PATHS", [temp_config_dir / "nonexistent.json"]):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("ANTHROPIC_API_KEY", None)
                key = load_api_key(None)
                assert key is None

    def test_save_and_load_key(self, temp_config_dir):
        """Save key to config file and load it back."""
        config_file = temp_config_dir / "config.json"
        saved_path = save_api_key("test-key-123", config_file)

        assert saved_path == config_file
        assert config_file.exists()

        # Verify content
        with open(config_file) as f:
            config = json.load(f)
        assert config["anthropic_api_key"] == "test-key-123"

    def test_save_preserves_existing_config(self, temp_config_dir):
        """Saving key preserves other config values."""
        config_file = temp_config_dir / "config.json"
        config_file.write_text('{"other_setting": "value"}')

        save_api_key("new-key", config_file)

        with open(config_file) as f:
            config = json.load(f)
        assert config["anthropic_api_key"] == "new-key"
        assert config["other_setting"] == "value"


# ---------------------------------------------------------------------------
# Unit Tests: ClaudeNLIConfig
# ---------------------------------------------------------------------------

class TestClaudeNLIConfig:
    """Test configuration dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ClaudeNLIConfig()
        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_tokens == 1024
        assert config.temperature == 0.0
        assert config.lang == "auto"

    def test_custom_values(self):
        """Test custom configuration."""
        config = ClaudeNLIConfig(
            model="claude-opus-4-20250514",
            max_tokens=2048,
            temperature=0.1,
            lang="fr"
        )
        assert config.model == "claude-opus-4-20250514"
        assert config.max_tokens == 2048

    def test_str_repr(self):
        """Test string representation."""
        config = ClaudeNLIConfig(model="test-model", lang="en")
        s = str(config)
        assert "test-model" in s
        assert "en" in s


# ---------------------------------------------------------------------------
# Unit Tests: ClaudeNLIClient (Mocked)
# ---------------------------------------------------------------------------

class TestClaudeNLIClientMocked:
    """Test Claude NLI client with mocked API."""

    def test_init_requires_key(self):
        """Initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with patch("RAGGAE.core.nli_claude.CONFIG_PATHS", []):
                with pytest.raises(ValueError, match="API key not found"):
                    ClaudeNLIClient()

    def test_check_single(self, mock_anthropic):
        """Test single clause-requirement check."""
        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"label": "Yes", "rationale": "Matches perfectly"}')]
        mock_anthropic.messages.create.return_value = mock_response

        client = ClaudeNLIClient(api_key="test-key")
        result = client.check(
            clause="The provider is ISO 27001 certified.",
            requirement="Provider must be ISO 27001 certified"
        )

        assert isinstance(result, NLIResult)
        assert result.label == "Yes"
        assert result.rationale == "Matches perfectly"

    def test_check_returns_no_on_error(self, mock_anthropic):
        """Test error handling returns No verdict."""
        mock_anthropic.messages.create.side_effect = Exception("API error")

        client = ClaudeNLIClient(api_key="test-key")
        result = client.check("clause", "requirement")

        assert result.label == "No"
        assert "API error" in result.rationale

    def test_check_batch(self, mock_anthropic):
        """Test batch processing."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='''[
            {"label": "Yes", "rationale": "First match"},
            {"label": "No", "rationale": "No match"},
            {"label": "Partial", "rationale": "Partial match"}
        ]''')]
        mock_anthropic.messages.create.return_value = mock_response

        client = ClaudeNLIClient(api_key="test-key")
        results = client.check_batch([
            ("clause1", "req1"),
            ("clause2", "req2"),
            ("clause3", "req3"),
        ])

        assert len(results) == 3
        assert results[0].label == "Yes"
        assert results[1].label == "No"
        assert results[2].label == "Partial"

    def test_check_batch_single_item(self, mock_anthropic):
        """Batch with single item uses check() directly."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"label": "Yes", "rationale": "OK"}')]
        mock_anthropic.messages.create.return_value = mock_response

        client = ClaudeNLIClient(api_key="test-key")
        results = client.check_batch([("clause", "req")])

        assert len(results) == 1
        assert results[0].label == "Yes"

    def test_check_batch_empty(self, mock_anthropic):
        """Empty batch returns empty list."""
        client = ClaudeNLIClient(api_key="test-key")
        results = client.check_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# Unit Tests: Factory Function
# ---------------------------------------------------------------------------

class TestCreateNLIClient:
    """Test the factory function."""

    def test_create_ollama_client(self):
        """Create Ollama client (default)."""
        with patch("RAGGAE.core.nli_claude.NLIClient") as mock_nli:
            client = create_nli_client(backend="ollama", model="mistral")
            mock_nli.assert_called_once()

    def test_create_claude_client(self, mock_anthropic):
        """Create Claude client."""
        client = create_nli_client(
            backend="claude",
            api_key="test-key",
            model="claude-sonnet-4-20250514"
        )
        assert isinstance(client, ClaudeNLIClient)

    def test_default_is_ollama(self):
        """Default backend is Ollama."""
        with patch("RAGGAE.core.nli_claude.NLIClient") as mock_nli:
            create_nli_client()
            mock_nli.assert_called_once()


# ---------------------------------------------------------------------------
# Integration Tests (require ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestClaudeNLIClientIntegration:
    """Integration tests requiring real API access.

    Run with: pytest tests/test_nli_claude.py -v -m integration

    Requires ANTHROPIC_API_KEY environment variable or config file.
    """

    @pytest.fixture
    def client(self):
        """Create real Claude client."""
        key = load_api_key()
        if not key:
            pytest.skip("ANTHROPIC_API_KEY not available")
        return ClaudeNLIClient(api_key=key, config=ClaudeNLIConfig(
            model="claude-haiku-3-5-20241022",  # Use fastest model for tests
            temperature=0.0
        ))

    def test_real_check_yes(self, client):
        """Real API call - should return Yes."""
        result = client.check(
            clause="Le prestataire est certifié ISO/IEC 27001:2022 par Bureau Veritas.",
            requirement="Provider must be ISO 27001 certified"
        )
        assert result.label in ("Yes", "Partial")
        assert len(result.rationale) > 0

    def test_real_check_no(self, client):
        """Real API call - should return No."""
        result = client.check(
            clause="We provide excellent customer service 24/7.",
            requirement="Provider must be ISO 27001 certified"
        )
        assert result.label in ("No", "Partial")

    def test_real_batch(self, client):
        """Real API batch call."""
        results = client.check_batch([
            ("ISO 27001 certified since 2020", "ISO 27001 certification"),
            ("Uses MLflow on Kubernetes", "MLflow platform required"),
            ("Office located in Paris", "GDPR compliance required"),
        ])
        assert len(results) == 3
        for r in results:
            assert r.label in ("Yes", "No", "Partial")
            assert len(r.rationale) > 0


# ---------------------------------------------------------------------------
# Pytest markers configuration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require API key)"
    )
