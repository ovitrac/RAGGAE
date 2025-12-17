# tests/conftest.py
"""
Pytest configuration and fixtures for RAGGAE tests.

Author: Dr. Olivier Vitrac, PhD, HDR
Organization: Adservio
Date: December 17, 2025
License: MIT
"""
import os
import tempfile
from pathlib import Path
import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require external services)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "ollama: marks tests requiring Ollama to be running"
    )
    config.addinivalue_line(
        "markers", "claude: marks tests requiring Claude API key"
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file for testing."""
    content = """
TENDER RESPONSE: Cloud Infrastructure Services

1. SECURITY CERTIFICATIONS

Our organization holds the following certifications:
- ISO/IEC 27001:2022 certified by Bureau Veritas
- SOC 2 Type II compliant
- GDPR compliant data processing

2. TECHNICAL CAPABILITIES

2.1 Container Orchestration
We provide enterprise Kubernetes clusters with:
- Auto-scaling and self-healing
- GitOps-based deployments using ArgoCD
- Service mesh with Istio

2.2 MLOps Platform
Our MLOps infrastructure includes:
- MLflow for experiment tracking and model registry
- Kubeflow for ML pipelines
- GPU support for training workloads

3. DATA SOVEREIGNTY

All data is processed and stored within EU data centers:
- Primary: Paris, France
- Backup: Frankfurt, Germany
- No data transfer outside EU without explicit consent
    """
    file_path = temp_dir / "sample_tender.txt"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def ollama_available():
    """Check if Ollama is running."""
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture
def claude_available():
    """Check if Claude API key is available."""
    from RAGGAE.core.nli_claude import load_api_key
    return load_api_key() is not None


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and available services."""
    skip_integration = pytest.mark.skip(reason="integration tests disabled")
    skip_ollama = pytest.mark.skip(reason="Ollama not running")
    skip_claude = pytest.mark.skip(reason="Claude API key not available")

    # Check if we should run integration tests
    run_integration = config.getoption("-m", default="") == "integration"

    for item in items:
        # Skip integration tests by default
        if "integration" in item.keywords and not run_integration:
            item.add_marker(skip_integration)
