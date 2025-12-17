"""Performance test fixtures and configuration.

Provides fixtures for performance benchmarking:
- Timer utilities
- Memory tracking
- Test document generation
- Mock services for consistent benchmarking
"""

from __future__ import annotations

import gc
import os
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@dataclass
class TimingResult:
    """Result of a timed operation."""

    operation: str
    duration_ms: float
    target_ms: float
    passed: bool

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.operation}: {self.duration_ms:.2f}ms (target: {self.target_ms}ms)"


@dataclass
class MemoryResult:
    """Result of memory measurement."""

    operation: str
    peak_mb: float
    current_mb: float
    target_mb: float
    passed: bool

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.operation}: peak={self.peak_mb:.2f}MB, current={self.current_mb:.2f}MB (target: {self.target_mb}MB)"


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, operation: str, target_ms: float):
        self.operation = operation
        self.target_ms = target_ms
        self.start_time: float = 0
        self.end_time: float = 0

    def __enter__(self) -> "PerformanceTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def passed(self) -> bool:
        return self.duration_ms <= self.target_ms

    @property
    def result(self) -> TimingResult:
        return TimingResult(
            operation=self.operation,
            duration_ms=self.duration_ms,
            target_ms=self.target_ms,
            passed=self.passed,
        )


class MemoryTracker:
    """Context manager for tracking memory usage."""

    def __init__(self, operation: str, target_mb: float):
        self.operation = operation
        self.target_mb = target_mb
        self.peak_mb: float = 0
        self.current_mb: float = 0

    def __enter__(self) -> "MemoryTracker":
        gc.collect()
        tracemalloc.start()
        return self

    def __exit__(self, *args) -> None:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.current_mb = current / (1024 * 1024)
        self.peak_mb = peak / (1024 * 1024)

    @property
    def passed(self) -> bool:
        return self.peak_mb <= self.target_mb

    @property
    def result(self) -> MemoryResult:
        return MemoryResult(
            operation=self.operation,
            peak_mb=self.peak_mb,
            current_mb=self.current_mb,
            target_mb=self.target_mb,
            passed=self.passed,
        )


@pytest.fixture
def performance_timer():
    """Factory fixture for creating performance timers."""

    def _create_timer(operation: str, target_ms: float) -> PerformanceTimer:
        return PerformanceTimer(operation, target_ms)

    return _create_timer


@pytest.fixture
def memory_tracker():
    """Factory fixture for creating memory trackers."""

    def _create_tracker(operation: str, target_mb: float) -> MemoryTracker:
        return MemoryTracker(operation, target_mb)

    return _create_tracker


@pytest.fixture
def test_documents() -> List[dict]:
    """Generate test documents for benchmarking."""
    documents = []
    for i in range(100):
        documents.append(
            {
                "id": f"doc-{i:04d}",
                "content": f"This is test document {i}. " * 50,
                "metadata": {
                    "source": "test",
                    "timestamp": f"2024-12-{(i % 28) + 1:02d}T10:00:00Z",
                    "type": "note",
                },
            }
        )
    return documents


@pytest.fixture
def mock_ollama():
    """Mock Ollama for consistent performance testing."""
    with patch("futurnal.search.hybrid.performance.ollama_pool.httpx") as mock_httpx:
        mock_client = AsyncMock()
        mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

        # Fast mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "This is a test response for benchmarking.",
            "done": True,
        }
        mock_client.post.return_value = mock_response

        yield mock_client


@pytest.fixture
def mock_embeddings():
    """Mock embedding service for consistent performance testing."""
    with patch("futurnal.embeddings.service.MultiModelEmbeddingService") as mock_svc:
        mock_instance = MagicMock()

        # Return fast mock embeddings
        async def mock_embed(text: str) -> List[float]:
            # Return consistent-length embedding vector
            return [0.1] * 384

        mock_instance.embed = AsyncMock(side_effect=mock_embed)
        mock_instance.embed_batch = AsyncMock(
            side_effect=lambda texts: [[0.1] * 384 for _ in texts]
        )
        mock_svc.return_value = mock_instance

        yield mock_instance


@pytest.fixture
def mock_pkg():
    """Mock PKG client for consistent performance testing."""
    with patch("futurnal.pkg.client.PKGClient") as mock_client:
        mock_instance = MagicMock()

        # Fast mock queries
        mock_instance.query = AsyncMock(return_value=[])
        mock_instance.search = AsyncMock(return_value=[])
        mock_instance.get_entity = AsyncMock(return_value=None)

        mock_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temporary data directory for tests."""
    data_dir = tmp_path / "futurnal_test"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


# Performance test markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "performance: marks tests as performance benchmarks",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow running",
    )
