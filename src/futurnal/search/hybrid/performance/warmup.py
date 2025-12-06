"""Model Warm-Up System for Cold Start Mitigation.

Manages model preloading to avoid cold start latency when making
LLM inference requests.

Key Features:
- Parallel warm-up of multiple models on startup
- Background keep-warm task to prevent model unloading
- Retry logic for warm-up failures
- Configurable warm-up prompts per model

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/05-performance-caching.md

Option B Compliance:
- Ghost model frozen, warm-up only loads models for inference
- Local-first processing via Ollama
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from futurnal.search.hybrid.performance.ollama_pool import OllamaConnectionPool

logger = logging.getLogger(__name__)


# Default warm-up prompts by model (minimal prompts for fast warm-up)
DEFAULT_WARMUP_PROMPTS: Dict[str, str] = {
    "llama3.1:8b": "Classify this query: 'test query'",
    "llama3.1:8b-instruct-q4_0": "Classify this query: 'test query'",
    "phi3:mini": "Intent: 'hello'",
    "qwen2.5-coder:32b-instruct-q4_0": "What is Python?",
    "bielik:4.5b-instruct-q4_0": "Cześć",
}


@dataclass
class WarmUpConfig:
    """Configuration for model warm-up."""

    models: List[str] = field(default_factory=lambda: ["llama3.1:8b"])
    warmup_prompts: Dict[str, str] = field(default_factory=lambda: DEFAULT_WARMUP_PROMPTS.copy())
    parallel_warmup: bool = True  # Warm up models in parallel
    timeout_seconds: float = 60.0  # Max time for warm-up per model
    retry_on_failure: bool = True
    keep_warm_interval_seconds: float = 60.0  # Ping interval to keep models loaded


class ModelWarmUpManager:
    """Manages model warm-up to avoid cold start latency.

    Cold start costs:
    - Ollama model load: 2-5 seconds
    - Embedding model load: 1-3 seconds
    - First inference: 0.5-1 second (compilation)

    Warm-up strategy:
    1. On application start: warm up primary models
    2. On idle: keep models warm with periodic pings
    3. On model switch: pre-warm next likely model

    Integration Points:
    - OllamaConnectionPool: Uses pool for warm-up requests
    - Application startup: Called during initialization

    Example:
        >>> pool = OllamaConnectionPool()
        >>> await pool.initialize()
        >>> warmup = ModelWarmUpManager(pool)
        >>> await warmup.warm_up_all()
        >>> await warmup.start_keep_warm()
    """

    def __init__(
        self,
        ollama_pool: OllamaConnectionPool,
        config: Optional[WarmUpConfig] = None,
    ) -> None:
        """Initialize model warm-up manager.

        Args:
            ollama_pool: OllamaConnectionPool for making warm-up requests
            config: Optional warm-up configuration
        """
        self.ollama_pool = ollama_pool
        self.config = config or WarmUpConfig()
        self._warm_models: Dict[str, float] = {}  # model -> last warm time
        self._keep_warm_task: Optional[asyncio.Task[None]] = None
        self._running = False

    async def warm_up_all(self) -> bool:
        """Warm up all configured models.

        Called during application startup.

        Returns:
            True if all models warmed successfully, False otherwise
        """
        logger.info(f"Starting model warm-up for: {self.config.models}")
        start_time = asyncio.get_event_loop().time()

        if self.config.parallel_warmup:
            tasks = [self._warm_up_model(model) for model in self.config.models]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Convert exceptions to False
            results = [r if isinstance(r, bool) else False for r in results]
        else:
            results: List[bool] = []
            for model in self.config.models:
                result = await self._warm_up_model(model)
                results.append(result)

        elapsed = asyncio.get_event_loop().time() - start_time
        success_count = sum(1 for r in results if r is True)

        logger.info(
            f"Model warm-up complete: {success_count}/{len(self.config.models)} "
            f"models warmed in {elapsed:.2f}s"
        )

        return success_count == len(self.config.models)

    async def warm_up_model(self, model: str) -> bool:
        """Warm up a specific model.

        Args:
            model: Model name to warm up

        Returns:
            True if successful, False otherwise
        """
        return await self._warm_up_model(model)

    async def _warm_up_model(self, model: str) -> bool:
        """Internal model warm-up implementation.

        Args:
            model: Model name to warm up

        Returns:
            True if successful, False otherwise
        """
        prompt = self.config.warmup_prompts.get(model, "Hello")

        try:
            await asyncio.wait_for(
                self.ollama_pool.generate(
                    model=model,
                    prompt=prompt,
                    num_predict=1,  # Minimal generation
                ),
                timeout=self.config.timeout_seconds,
            )
            self._warm_models[model] = asyncio.get_event_loop().time()
            logger.debug(f"Model {model} warmed up successfully")
            return True

        except asyncio.TimeoutError:
            logger.warning(f"Warm-up timeout for model {model}")
            if self.config.retry_on_failure:
                return await self._retry_warmup(model, prompt)
            return False

        except Exception as e:
            logger.warning(f"Failed to warm up model {model}: {e}")
            if self.config.retry_on_failure:
                return await self._retry_warmup(model, prompt)
            return False

    async def _retry_warmup(self, model: str, prompt: str) -> bool:
        """Retry warm-up after initial failure.

        Args:
            model: Model name
            prompt: Warm-up prompt

        Returns:
            True if retry successful, False otherwise
        """
        await asyncio.sleep(1.0)
        try:
            await asyncio.wait_for(
                self.ollama_pool.generate(
                    model=model,
                    prompt=prompt,
                    num_predict=1,
                ),
                timeout=self.config.timeout_seconds,
            )
            self._warm_models[model] = asyncio.get_event_loop().time()
            logger.debug(f"Model {model} warmed up on retry")
            return True
        except Exception as e:
            logger.error(f"Warm-up retry failed for model {model}: {e}")
            return False

    async def start_keep_warm(
        self,
        interval_seconds: Optional[float] = None,
    ) -> None:
        """Start background task to keep models warm.

        Args:
            interval_seconds: Optional override for keep-warm interval
        """
        if self._running:
            logger.warning("Keep-warm task already running")
            return

        interval = interval_seconds or self.config.keep_warm_interval_seconds
        self._running = True
        self._keep_warm_task = asyncio.create_task(self._keep_warm_loop(interval))
        logger.info(f"Started keep-warm task with {interval}s interval")

    async def stop_keep_warm(self) -> None:
        """Stop the keep-warm background task."""
        self._running = False
        if self._keep_warm_task:
            self._keep_warm_task.cancel()
            try:
                await self._keep_warm_task
            except asyncio.CancelledError:
                pass
            self._keep_warm_task = None
            logger.info("Stopped keep-warm task")

    async def _keep_warm_loop(self, interval: float) -> None:
        """Periodic ping to keep models loaded.

        Args:
            interval: Seconds between pings
        """
        while self._running:
            await asyncio.sleep(interval)

            for model in self.config.models:
                if model in self._warm_models:
                    try:
                        await self.ollama_pool.generate(
                            model=model,
                            prompt="ping",
                            num_predict=1,
                        )
                        self._warm_models[model] = asyncio.get_event_loop().time()
                        logger.debug(f"Keep-warm ping for {model}")
                    except Exception:
                        # Silent failure for keep-warm
                        pass

    def is_warm(self, model: str) -> bool:
        """Check if a model is currently warm.

        Args:
            model: Model name to check

        Returns:
            True if model was recently warmed
        """
        return model in self._warm_models

    def get_warm_models(self) -> List[str]:
        """Get list of currently warm models.

        Returns:
            List of model names that have been warmed
        """
        return list(self._warm_models.keys())

    def get_status(self) -> Dict[str, Any]:
        """Get warm-up manager status.

        Returns:
            Dict with status information
        """
        return {
            "warm_models": list(self._warm_models.keys()),
            "keep_warm_running": self._running,
            "configured_models": self.config.models,
        }
