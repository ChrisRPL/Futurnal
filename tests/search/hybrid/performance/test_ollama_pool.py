"""Tests for OllamaConnectionPool.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/05-performance-caching.md
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Check if aiohttp is available
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from futurnal.search.hybrid.performance.ollama_pool import (
    HealthStatus,
    OllamaConnectionConfig,
    OllamaConnectionError,
)

# Conditionally import pool if aiohttp available
if AIOHTTP_AVAILABLE:
    from futurnal.search.hybrid.performance.ollama_pool import OllamaConnectionPool


class TestHealthStatus:
    """Tests for HealthStatus circuit breaker."""

    def test_initial_state(self, health_status: HealthStatus):
        """Test initial healthy state."""
        assert health_status.is_healthy is True
        assert health_status.consecutive_failures == 0
        assert health_status.consecutive_successes == 0

    def test_record_success(self, health_status: HealthStatus):
        """Test recording successful requests."""
        health_status.record_success()
        assert health_status.consecutive_successes == 1
        assert health_status.consecutive_failures == 0

    def test_record_failure(self, health_status: HealthStatus):
        """Test recording failed requests."""
        health_status.record_failure()
        assert health_status.consecutive_failures == 1
        assert health_status.consecutive_successes == 0

    def test_circuit_breaker_opens(self, health_status: HealthStatus):
        """Test circuit breaker opens after threshold failures."""
        for _ in range(5):  # Default threshold
            health_status.record_failure()

        assert health_status.is_healthy is False

    def test_circuit_breaker_closes(self, health_status: HealthStatus):
        """Test circuit breaker closes after recovery threshold."""
        health_status.mark_unhealthy()
        assert health_status.is_healthy is False

        for _ in range(3):  # Default recovery threshold
            health_status.record_success()

        assert health_status.is_healthy is True


class TestOllamaConnectionConfig:
    """Tests for OllamaConnectionConfig."""

    def test_default_config(self, ollama_config: OllamaConnectionConfig):
        """Test default configuration values."""
        assert ollama_config.base_url == "http://localhost:11434"
        assert ollama_config.pool_size >= 1
        assert ollama_config.request_timeout > 0
        assert ollama_config.max_retries >= 1


@pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not available")
class TestOllamaConnectionPool:
    """Tests for OllamaConnectionPool."""

    @pytest.mark.asyncio
    async def test_initialization(self, ollama_config: OllamaConnectionConfig):
        """Test pool initialization."""
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(return_value=AsyncMock(
                status=200,
                raise_for_status=MagicMock(),
            ))
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            pool = OllamaConnectionPool(ollama_config)
            await pool.initialize()

            assert pool.is_initialized is True

            await pool.close()
            assert pool.is_initialized is False

    @pytest.mark.asyncio
    async def test_generate(self, ollama_config: OllamaConnectionConfig):
        """Test text generation."""
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json = AsyncMock(return_value={"response": "Generated text"})
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock()

            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.get = AsyncMock(return_value=AsyncMock(
                status=200,
                raise_for_status=MagicMock(),
            ))
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            pool = OllamaConnectionPool(ollama_config)
            await pool.initialize()

            result = await pool.generate("llama3.1:8b", "Hello")

            assert result == "Generated text"
            await pool.close()

    @pytest.mark.asyncio
    async def test_batch_generate(self, ollama_config: OllamaConnectionConfig):
        """Test batch generation."""
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json = AsyncMock(return_value={"response": "Response"})
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock()

            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.get = AsyncMock(return_value=AsyncMock(
                status=200,
                raise_for_status=MagicMock(),
            ))
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            pool = OllamaConnectionPool(ollama_config)
            await pool.initialize()

            results = await pool.batch_generate("test-model", ["prompt1", "prompt2"])

            assert len(results) == 2
            await pool.close()

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, ollama_config: OllamaConnectionConfig):
        """Test retry logic on connection failure."""
        ollama_config.max_retries = 2
        ollama_config.retry_delay = 0.01  # Fast retry for testing

        with patch("aiohttp.ClientSession") as mock_session_cls:
            # First call fails, second succeeds
            mock_response_success = AsyncMock()
            mock_response_success.status = 200
            mock_response_success.raise_for_status = MagicMock()
            mock_response_success.json = AsyncMock(return_value={"response": "OK"})
            mock_response_success.__aenter__ = AsyncMock(return_value=mock_response_success)
            mock_response_success.__aexit__ = AsyncMock()

            mock_response_fail = AsyncMock()
            mock_response_fail.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("Connection error"))
            mock_response_fail.__aexit__ = AsyncMock()

            mock_session = AsyncMock()
            mock_session.post = MagicMock(side_effect=[mock_response_fail, mock_response_success])
            mock_session.get = AsyncMock(return_value=AsyncMock(
                status=200,
                raise_for_status=MagicMock(),
            ))
            mock_session.close = AsyncMock()
            mock_session_cls.return_value = mock_session

            pool = OllamaConnectionPool(ollama_config)
            await pool.initialize()

            result = await pool.generate("test-model", "test")
            assert result == "OK"

            await pool.close()

    def test_health_status_getter(self, ollama_config: OllamaConnectionConfig):
        """Test health status reporting."""
        pool = OllamaConnectionPool(ollama_config)
        status = pool.get_health_status()

        assert "is_healthy" in status
        assert "is_initialized" in status
        assert "consecutive_failures" in status
