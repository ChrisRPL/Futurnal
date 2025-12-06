"""Integration Test Fixtures for Hybrid Search API.

Provides fixtures for end-to-end integration testing:
- HybridSearchAPI instances
- Ollama connectivity checks
- Golden query loading
- Skip markers for infrastructure dependencies

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/06-integration-testing.md
"""

from __future__ import annotations

import os
import asyncio
from typing import Any, Dict, List, Optional

import pytest

from futurnal.search.api import HybridSearchAPI, create_hybrid_search_api
from futurnal.search.config import SearchConfig

# Import golden queries
from tests.search.fixtures.golden_queries import (
    load_golden_query_set,
    generate_benchmark_queries,
)


# ---------------------------------------------------------------------------
# Skip Markers
# ---------------------------------------------------------------------------

def _check_ollama_available() -> bool:
    """Check if Ollama server is available."""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def _check_neo4j_available() -> bool:
    """Check if Neo4j is available via testcontainers."""
    try:
        from testcontainers.neo4j import Neo4jContainer
        return True
    except ImportError:
        return False


def _check_docker_available() -> bool:
    """Check if Docker is available."""
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


OLLAMA_AVAILABLE = _check_ollama_available()
NEO4J_AVAILABLE = _check_neo4j_available()
DOCKER_AVAILABLE = _check_docker_available()

requires_ollama = pytest.mark.skipif(
    not OLLAMA_AVAILABLE,
    reason="Ollama server not available at localhost:11434"
)

requires_neo4j = pytest.mark.skipif(
    not NEO4J_AVAILABLE,
    reason="Neo4j testcontainers not available"
)

requires_docker = pytest.mark.skipif(
    not DOCKER_AVAILABLE,
    reason="Docker not available"
)


# ---------------------------------------------------------------------------
# API Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def search_config() -> SearchConfig:
    """Provide default search configuration."""
    return SearchConfig()


@pytest.fixture
def api(search_config: SearchConfig) -> HybridSearchAPI:
    """Create HybridSearchAPI instance for testing."""
    return create_hybrid_search_api(
        config=search_config,
        caching_enabled=True,
    )


@pytest.fixture
def api_with_multimodal(search_config: SearchConfig) -> HybridSearchAPI:
    """Create HybridSearchAPI with multimodal support."""
    return create_hybrid_search_api(
        config=search_config,
        multimodal_enabled=True,
        caching_enabled=True,
    )


@pytest.fixture
def api_with_grpo(search_config: SearchConfig) -> HybridSearchAPI:
    """Create HybridSearchAPI with experiential learning."""
    return create_hybrid_search_api(
        config=search_config,
        experiential_learning=True,
        caching_enabled=True,
    )


@pytest.fixture
def api_no_cache(search_config: SearchConfig) -> HybridSearchAPI:
    """Create HybridSearchAPI without caching."""
    return create_hybrid_search_api(
        config=search_config,
        caching_enabled=False,
    )


# ---------------------------------------------------------------------------
# Golden Query Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def golden_queries() -> List[Dict[str, Any]]:
    """Load all golden queries."""
    return load_golden_query_set()


@pytest.fixture
def temporal_queries() -> List[Dict[str, Any]]:
    """Load temporal golden queries."""
    return load_golden_query_set(query_type="temporal")


@pytest.fixture
def causal_queries() -> List[Dict[str, Any]]:
    """Load causal golden queries."""
    return load_golden_query_set(query_type="causal")


@pytest.fixture
def ocr_queries() -> List[Dict[str, Any]]:
    """Load OCR golden queries."""
    return load_golden_query_set(modality="ocr")


@pytest.fixture
def audio_queries() -> List[Dict[str, Any]]:
    """Load audio golden queries."""
    return load_golden_query_set(modality="audio")


@pytest.fixture
def benchmark_queries() -> List[str]:
    """Generate benchmark queries for performance testing."""
    return generate_benchmark_queries(n=100)


# ---------------------------------------------------------------------------
# Async Helper
# ---------------------------------------------------------------------------

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
