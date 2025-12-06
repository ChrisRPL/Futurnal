"""Tests for ModelWarmUpManager.

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/05-performance-caching.md
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from futurnal.search.hybrid.performance.warmup import (
    ModelWarmUpManager,
    WarmUpConfig,
    DEFAULT_WARMUP_PROMPTS,
)


class TestWarmUpConfig:
    """Tests for WarmUpConfig."""

    def test_default_config(self, warmup_config: WarmUpConfig):
        """Test default configuration values."""
        assert len(warmup_config.models) > 0
        assert warmup_config.parallel_warmup is True
        assert warmup_config.timeout_seconds > 0

    def test_default_warmup_prompts(self):
        """Verify default warm-up prompts are defined."""
        assert "llama3.1:8b" in DEFAULT_WARMUP_PROMPTS


class TestModelWarmUpManager:
    """Tests for ModelWarmUpManager."""

    @pytest.mark.asyncio
    async def test_warm_up_single_model(
        self, mock_ollama_pool: MagicMock, warmup_config: WarmUpConfig
    ):
        """Test warming up a single model."""
        manager = ModelWarmUpManager(mock_ollama_pool, warmup_config)

        result = await manager.warm_up_model("test-model")

        assert result is True
        mock_ollama_pool.generate.assert_called()

    @pytest.mark.asyncio
    async def test_warm_up_all_parallel(
        self, mock_ollama_pool: MagicMock
    ):
        """Test parallel warm-up of multiple models."""
        config = WarmUpConfig(
            models=["model1", "model2", "model3"],
            warmup_prompts={"model1": "p1", "model2": "p2", "model3": "p3"},
            parallel_warmup=True,
            timeout_seconds=5.0,
        )

        manager = ModelWarmUpManager(mock_ollama_pool, config)
        result = await manager.warm_up_all()

        assert result is True
        assert mock_ollama_pool.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_warm_up_all_sequential(self, mock_ollama_pool: MagicMock):
        """Test sequential warm-up of multiple models."""
        config = WarmUpConfig(
            models=["model1", "model2"],
            warmup_prompts={"model1": "p1", "model2": "p2"},
            parallel_warmup=False,
            timeout_seconds=5.0,
        )

        manager = ModelWarmUpManager(mock_ollama_pool, config)
        result = await manager.warm_up_all()

        assert result is True

    @pytest.mark.asyncio
    async def test_warm_up_tracks_models(
        self, mock_ollama_pool: MagicMock, warmup_config: WarmUpConfig
    ):
        """Test that warm models are tracked."""
        manager = ModelWarmUpManager(mock_ollama_pool, warmup_config)

        await manager.warm_up_model("test-model")

        assert manager.is_warm("test-model") is True
        assert "test-model" in manager.get_warm_models()

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, warmup_config: WarmUpConfig):
        """Test retry logic on warm-up failure."""
        mock_pool = MagicMock()
        mock_pool.generate = AsyncMock(side_effect=[Exception("Fail"), "Success"])

        warmup_config.retry_on_failure = True

        manager = ModelWarmUpManager(mock_pool, warmup_config)
        result = await manager.warm_up_model("test-model")

        assert result is True
        assert mock_pool.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_handling(self, warmup_config: WarmUpConfig):
        """Test timeout handling during warm-up."""
        mock_pool = MagicMock()

        async def slow_generate(*args, **kwargs):
            import asyncio
            await asyncio.sleep(10)  # Longer than timeout
            return "result"

        mock_pool.generate = slow_generate
        warmup_config.timeout_seconds = 0.1
        warmup_config.retry_on_failure = False

        manager = ModelWarmUpManager(mock_pool, warmup_config)
        result = await manager.warm_up_model("test-model")

        assert result is False

    @pytest.mark.asyncio
    async def test_start_stop_keep_warm(
        self, mock_ollama_pool: MagicMock, warmup_config: WarmUpConfig
    ):
        """Test keep-warm task lifecycle."""
        import asyncio

        manager = ModelWarmUpManager(mock_ollama_pool, warmup_config)

        # Warm up first so models are tracked
        await manager.warm_up_all()

        # Start keep-warm
        await manager.start_keep_warm(interval_seconds=0.1)

        # Wait a bit for at least one ping
        await asyncio.sleep(0.2)

        # Stop keep-warm
        await manager.stop_keep_warm()

        status = manager.get_status()
        assert status["keep_warm_running"] is False

    def test_get_status(
        self, mock_ollama_pool: MagicMock, warmup_config: WarmUpConfig
    ):
        """Test status reporting."""
        manager = ModelWarmUpManager(mock_ollama_pool, warmup_config)

        status = manager.get_status()

        assert "warm_models" in status
        assert "keep_warm_running" in status
        assert "configured_models" in status
