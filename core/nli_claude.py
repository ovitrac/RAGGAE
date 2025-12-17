"""
RAGGAE/core/nli_claude.py

Natural Language Inference client using Claude API for compliance checking.
Alternative to Ollama for developers without local GPU resources.

This module provides:
- ClaudeNLIClient: Claude-based NLI with batch optimization
- NLIResult: Shared dataclass for inference results (from nli_ollama)
- ClaudeNLIConfig: Configuration for Claude model and inference parameters
- load_api_key(): Helper to load API key from multiple sources

Key Features:
- Batch processing: Multiple clause/requirement pairs in single API call
- Same interface as OllamaNLIClient for drop-in replacement
- Flexible API key resolution: explicit > env var > config file

API Key Resolution (in priority order):
1. Explicit api_key parameter
2. ANTHROPIC_API_KEY environment variable
3. Config file: ~/.config/raggae/config.json
4. Config file: ~/.raggae.json

Config file format (JSON):
{
    "anthropic_api_key": "sk-ant-..."
}

Author: Dr. Olivier Vitrac, PhD, HDR
Email: olivier.vitrac@adservio.com
Organization: Adservio
Date: December 17, 2025
License: MIT

Examples
--------
>>> from core.nli_claude import ClaudeNLIClient, ClaudeNLIConfig
>>> # With explicit key
>>> nli = ClaudeNLIClient(api_key="sk-ant-...")
>>>
>>> # With env var or config file (auto-resolved)
>>> nli = ClaudeNLIClient()
>>>
>>> result = nli.check(
...     clause="Le prestataire est certifié ISO/IEC 27001:2022.",
...     requirement="Provider must be ISO 27001 certified"
... )
>>> print(f"{result.label}: {result.rationale}")

# Batch processing (optimized)
>>> results = nli.check_batch([
...     ("Clause 1", "Requirement 1"),
...     ("Clause 2", "Requirement 2"),
... ])
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import os
import re

# Import shared NLIResult from ollama module
from .nli_ollama import NLIResult, ALLOWED, _sanitize_label, _looks_invalid

# Config file locations (in priority order)
CONFIG_PATHS = [
    Path.home() / ".config" / "raggae" / "config.json",
    Path.home() / ".raggae.json",
]


def load_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """
    Load Anthropic API key from multiple sources.

    Resolution order:
    1. Explicit key parameter (if provided)
    2. ANTHROPIC_API_KEY environment variable
    3. Config file (~/.config/raggae/config.json)
    4. Config file (~/.raggae.json)

    Parameters
    ----------
    explicit_key : str, optional
        Explicitly provided API key (highest priority).

    Returns
    -------
    str or None
        API key if found, None otherwise.

    Examples
    --------
    >>> key = load_api_key()  # Auto-resolve from env/config
    >>> key = load_api_key("sk-ant-...")  # Use explicit key
    """
    # 1. Explicit key
    if explicit_key:
        return explicit_key

    # 2. Environment variable
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return env_key

    # 3. Config files
    for config_path in CONFIG_PATHS:
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                key = config.get("anthropic_api_key")
                if key:
                    return key
            except (json.JSONDecodeError, IOError, KeyError):
                continue  # Try next config file

    return None


def get_config_path() -> Path:
    """
    Get the recommended config file path.

    Returns the XDG-compliant path (~/.config/raggae/config.json).
    Creates the directory if it doesn't exist.

    Returns
    -------
    Path
        Path to the config file (may not exist yet).
    """
    config_dir = Path.home() / ".config" / "raggae"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.json"


def save_api_key(api_key: str, config_path: Optional[Path] = None) -> Path:
    """
    Save API key to config file.

    Parameters
    ----------
    api_key : str
        The Anthropic API key to save.
    config_path : Path, optional
        Custom config file path. Defaults to ~/.config/raggae/config.json.

    Returns
    -------
    Path
        Path where the config was saved.

    Example
    -------
    >>> save_api_key("sk-ant-...")
    PosixPath('/home/user/.config/raggae/config.json')
    """
    if config_path is None:
        config_path = get_config_path()

    # Load existing config or create new
    config = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    config["anthropic_api_key"] = api_key

    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Set restrictive permissions (owner read/write only)
    try:
        config_path.chmod(0o600)
    except OSError:
        pass  # Windows may not support chmod

    return config_path

# Lazy import for anthropic
_anthropic = None

def _get_anthropic():
    """Lazy import of anthropic SDK."""
    global _anthropic
    if _anthropic is None:
        try:
            import anthropic
            _anthropic = anthropic
        except ImportError:
            raise ImportError(
                "anthropic SDK not installed. Install with: pip install anthropic"
            )
    return _anthropic


@dataclass
class ClaudeNLIConfig:
    """Configuration for Claude NLI client.

    Attributes
    ----------
    model : str
        Claude model to use. Options:
        - "claude-sonnet-4-20250514" (recommended: fast + capable)
        - "claude-opus-4-20250514" (highest quality)
        - "claude-haiku-3-5-20241022" (fastest, lowest cost)
    max_tokens : int
        Maximum tokens in response.
    temperature : float
        Sampling temperature (0.0 for deterministic).
    lang : str
        Language hint for prompts ("fr", "en", "auto").
    """
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.0
    lang: str = "auto"

    def __str__(self) -> str:
        return f"ClaudeNLIConfig(model={self.model}, temp={self.temperature}, lang={self.lang})"


# System prompt for NLI
SYSTEM_PROMPT = """You are a strict compliance checker evaluating whether document clauses satisfy requirements.

For each clause-requirement pair, determine:
- "Yes": The clause clearly satisfies the requirement
- "No": The clause does not satisfy the requirement
- "Partial": The clause partially satisfies or is related but incomplete

Return ONLY valid JSON. Be concise but precise in rationales."""


def _parse_json_response(text: str) -> Optional[Dict]:
    """Parse JSON from Claude response, handling markdown code blocks."""
    text = text.strip()
    # Remove markdown code blocks
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.M)
    # Find JSON object or array
    match = re.search(r"[\[{].*[\]}]", text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class ClaudeNLIClient:
    """
    Claude-based NLI client for requirement satisfaction checks.

    Alternative to OllamaNLIClient for users without local GPU.
    API key is resolved from multiple sources (see load_api_key()).

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key. If not provided, resolved from:
        1. ANTHROPIC_API_KEY environment variable
        2. ~/.config/raggae/config.json
        3. ~/.raggae.json
    config : ClaudeNLIConfig, optional
        Configuration for model and inference parameters.

    Example
    -------
    >>> from core.nli_claude import ClaudeNLIClient
    >>> # Explicit key
    >>> nli = ClaudeNLIClient(api_key="sk-ant-...")
    >>>
    >>> # Auto-resolve from env/config
    >>> nli = ClaudeNLIClient()
    >>>
    >>> result = nli.check(
    ...     "Le prestataire est certifié ISO/IEC 27001:2022.",
    ...     "Provider must be ISO 27001 certified"
    ... )
    >>> print(result.label)  # "Yes"

    Batch processing (more efficient):
    >>> pairs = [
    ...     ("Uses MLflow on K8s", "MLflow required"),
    ...     ("Data in EU region", "GDPR compliance"),
    ... ]
    >>> results = nli.check_batch(pairs)
    """

    def __init__(self, api_key: Optional[str] = None, config: Optional[ClaudeNLIConfig] = None):
        # Resolve API key from multiple sources
        resolved_key = load_api_key(api_key)
        if not resolved_key:
            raise ValueError(
                "Anthropic API key not found. Provide via:\n"
                "  1. api_key parameter\n"
                "  2. ANTHROPIC_API_KEY environment variable\n"
                "  3. Config file: ~/.config/raggae/config.json\n"
                "     {\"anthropic_api_key\": \"sk-ant-...\"}"
            )

        anthropic = _get_anthropic()
        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.config = config or ClaudeNLIConfig()

    def check(self, clause: str, requirement: str) -> NLIResult:
        """
        Check if a clause satisfies a requirement.

        Parameters
        ----------
        clause : str
            The document clause/text to evaluate.
        requirement : str
            The requirement to check against.

        Returns
        -------
        NLIResult
            Result with label ("Yes", "No", "Partial") and rationale.
        """
        prompt = self._build_single_prompt(clause, requirement)

        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text
            parsed = _parse_json_response(content)

            if parsed is None:
                return NLIResult(
                    label="No",
                    rationale="Failed to parse JSON response from Claude"
                )

            label = _sanitize_label(parsed.get("label", "No"))
            rationale = str(parsed.get("rationale", "")).strip()

            if _looks_invalid(rationale):
                rationale = "Claude response had invalid/empty rationale."

            return NLIResult(label=label, rationale=rationale)

        except Exception as e:
            return NLIResult(
                label="No",
                rationale=f"Claude API error: {str(e)}"
            )

    def check_batch(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        """
        Check multiple clause-requirement pairs in a single API call.

        More efficient than calling check() multiple times.

        Parameters
        ----------
        pairs : List[Tuple[str, str]]
            List of (clause, requirement) tuples.

        Returns
        -------
        List[NLIResult]
            Results in same order as input pairs.
        """
        if not pairs:
            return []

        # For small batches, use single call
        if len(pairs) == 1:
            return [self.check(pairs[0][0], pairs[0][1])]

        prompt = self._build_batch_prompt(pairs)

        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens * len(pairs),  # Scale with batch size
                temperature=self.config.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text
            parsed = _parse_json_response(content)

            if parsed is None or not isinstance(parsed, list):
                # Fallback: parse as single responses
                return self._fallback_sequential(pairs)

            results = []
            for i, item in enumerate(parsed):
                if isinstance(item, dict):
                    label = _sanitize_label(item.get("label", "No"))
                    rationale = str(item.get("rationale", "")).strip()
                    if _looks_invalid(rationale):
                        rationale = "Invalid rationale in batch response."
                    results.append(NLIResult(label=label, rationale=rationale))
                else:
                    results.append(NLIResult(label="No", rationale="Invalid batch item"))

            # Ensure we have results for all pairs
            while len(results) < len(pairs):
                results.append(NLIResult(label="No", rationale="Missing from batch response"))

            return results[:len(pairs)]

        except Exception as e:
            # Fallback to sequential on error
            return self._fallback_sequential(pairs)

    def _fallback_sequential(self, pairs: List[Tuple[str, str]]) -> List[NLIResult]:
        """Fallback to sequential processing if batch fails."""
        return [self.check(clause, req) for clause, req in pairs]

    def _build_single_prompt(self, clause: str, requirement: str) -> str:
        """Build prompt for single clause-requirement check."""
        lang_hint = f"Language: {self.config.lang}. " if self.config.lang != "auto" else ""
        return f"""{lang_hint}Evaluate if this clause satisfies the requirement.

Clause: "{clause}"

Requirement: "{requirement}"

Respond with JSON: {{"label": "Yes|No|Partial", "rationale": "brief explanation"}}"""

    def _build_batch_prompt(self, pairs: List[Tuple[str, str]]) -> str:
        """Build prompt for batch clause-requirement checks."""
        lang_hint = f"Language: {self.config.lang}. " if self.config.lang != "auto" else ""

        items = []
        for i, (clause, req) in enumerate(pairs):
            items.append(f"""Item {i + 1}:
Clause: "{clause}"
Requirement: "{req}" """)

        items_text = "\n\n".join(items)

        return f"""{lang_hint}Evaluate each clause-requirement pair below.

{items_text}

Respond with a JSON array of objects, one per item, in order:
[
  {{"label": "Yes|No|Partial", "rationale": "brief explanation"}},
  ...
]"""

    # Compatibility with OllamaNLIClient interface
    def matrix(self, clauses: List[str], requirements: List[str], topk: int = 5):
        """
        Evaluate all clause-requirement combinations.

        Uses batch processing for efficiency.

        Parameters
        ----------
        clauses : List[str]
            Document clauses to evaluate.
        requirements : List[str]
            Requirements to check.
        topk : int
            Maximum clauses to check per requirement.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            DataFrame with all evaluations and Series with verdicts per requirement.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "`pandas` required for matrix(); install via `pip install pandas`"
            )

        clauses_list = list(clauses)[:topk]

        # Build all pairs
        pairs = []
        pair_info = []  # Track (req, hit_rank) for each pair
        for req in requirements:
            for i, clause in enumerate(clauses_list):
                pairs.append((clause, req))
                pair_info.append((req, i + 1, clause))

        # Batch process all pairs
        results = self.check_batch(pairs)

        # Build DataFrame
        rows = []
        for (req, rank, clause), result in zip(pair_info, results):
            rows.append({
                "requirement": req,
                "hit_rank": rank,
                "label": result.label,
                "rationale": result.rationale,
                "snippet": clause[:160].replace("\n", " ")
            })

        df = pd.DataFrame(rows)

        # Compute verdicts (best label per requirement)
        order = {"Yes": 2, "Partial": 1, "No": 0}
        verdict = (
            df.assign(score=df["label"].map(order))
            .groupby("requirement")["score"]
            .max()
            .map({2: "Yes", 1: "Partial", 0: "No"})
        )

        return df, verdict

    def __str__(self) -> str:
        return f"ClaudeNLIClient<{self.config}>"

    def __repr__(self) -> str:
        return f"ClaudeNLIClient(model={self.config.model!r}, lang={self.config.lang!r})"


# Factory function for easy switching between backends
def create_nli_client(
    backend: str = "ollama",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    lang: str = "auto",
    **kwargs
) -> Any:
    """
    Factory function to create NLI client based on backend choice.

    Parameters
    ----------
    backend : str
        "ollama" (default, local) or "claude" (API-based).
    api_key : str, optional
        For Claude backend. If not provided, resolved from:
        - ANTHROPIC_API_KEY environment variable
        - ~/.config/raggae/config.json
        - ~/.raggae.json
    model : str, optional
        Model name. Defaults:
        - Ollama: "mistral"
        - Claude: "claude-sonnet-4-20250514"
    lang : str
        Language hint ("fr", "en", "auto").
    **kwargs
        Additional config options passed to config class.

    Returns
    -------
    NLIClient or ClaudeNLIClient
        Configured NLI client.

    Example
    -------
    >>> # Local Ollama (default)
    >>> nli = create_nli_client()

    >>> # Claude API with explicit key
    >>> nli = create_nli_client(backend="claude", api_key="sk-ant-...")

    >>> # Claude API with auto-resolved key (env/config)
    >>> nli = create_nli_client(backend="claude")
    """
    if backend.lower() == "claude":
        # API key will be resolved by ClaudeNLIClient via load_api_key()
        config = ClaudeNLIConfig(
            model=model or "claude-sonnet-4-20250514",
            lang=lang,
            **{k: v for k, v in kwargs.items() if k in ("max_tokens", "temperature")}
        )
        return ClaudeNLIClient(api_key=api_key, config=config)

    else:  # ollama (default)
        from .nli_ollama import NLIClient, NLIConfig

        config = NLIConfig(
            model=model or "mistral",
            lang=lang,
            **{k: v for k, v in kwargs.items() if k in ("temperature", "num_ctx")}
        )
        return NLIClient(config=config)
