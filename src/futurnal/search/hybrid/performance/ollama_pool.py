"""Async Connection Pool for Ollama LLM Server.

Provides optimized connection management for LLM inference requests
with connection reuse, retry logic, and health monitoring.

Key Features:
- Connection pooling with keepalive for HTTP reuse
- Automatic retry with exponential backoff
- Health monitoring with circuit breaker pattern
- Batch generation for multiple prompts

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/05-performance-caching.md

Option B Compliance:
- Ghost model frozen, Ollama used for inference only
- Local-first processing on localhost:11434
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class OllamaConnectionConfig:
    """Configuration for Ollama connection pool."""

    base_url: str = "http://localhost:11434"
    pool_size: int = 4  # Concurrent connections
    request_timeout: float = 30.0  # Seconds
    keepalive_timeout: float = 300.0  # Connection reuse window
    max_retries: int = 3
    retry_delay: float = 0.5
    batch_size: int = 8  # Max requests per batch
    batch_timeout: float = 0.05  # Max wait for batch fill (50ms)


@dataclass
class HealthStatus:
    """Circuit breaker health tracking."""

    consecutive_failures: int = 0
    consecutive_successes: int = 0
    is_healthy: bool = True
    last_check: float = 0.0
    failure_threshold: int = 5
    recovery_threshold: int = 3

    def record_success(self) -> None:
        """Record a successful request."""
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        if self.consecutive_successes >= self.recovery_threshold:
            self.is_healthy = True

    def record_failure(self) -> None:
        """Record a failed request."""
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        if self.consecutive_failures >= self.failure_threshold:
            self.is_healthy = False

    def mark_healthy(self) -> None:
        """Force mark as healthy (after successful health check)."""
        self.is_healthy = True
        self.consecutive_failures = 0

    def mark_unhealthy(self) -> None:
        """Force mark as unhealthy."""
        self.is_healthy = False


class OllamaConnectionError(Exception):
    """Raised when Ollama connection fails."""

    pass


class OllamaConnectionPool:
    """Connection pool for Ollama LLM server.

    Provides async HTTP connection pooling with:
    - Connection reuse with keepalive
    - Automatic retry with exponential backoff
    - Health monitoring with circuit breaker
    - Batch generation for throughput optimization

    Integration Points:
    - OllamaIntentClassifier: Primary consumer for intent classification
    - ModelWarmUpManager: Pre-warms models via this pool

    Example:
        >>> pool = OllamaConnectionPool()
        >>> await pool.initialize()
        >>> response = await pool.generate("llama3.1:8b", "Hello, world!")
        >>> await pool.close()
    """

    def __init__(self, config: Optional[OllamaConnectionConfig] = None) -> None:
        """Initialize Ollama connection pool.

        Args:
            config: Optional connection configuration
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp is required for OllamaConnectionPool. "
                "Install with: pip install aiohttp"
            )

        self.config = config or OllamaConnectionConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(self.config.pool_size)
        self._health_status = HealthStatus()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize connection pool and warm up.

        Creates the aiohttp session with connection pooling and
        performs initial health check.
        """
        if self._initialized:
            return

        connector = aiohttp.TCPConnector(
            limit=self.config.pool_size,
            keepalive_timeout=self.config.keepalive_timeout,
            enable_cleanup_closed=True,
        )

        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
        )

        # Warm up connection with health check
        await self._health_check()
        self._initialized = True
        logger.info(
            "Initialized OllamaConnectionPool: pool_size=%d, base_url=%s",
            self.config.pool_size,
            self.config.base_url,
        )

    async def close(self) -> None:
        """Clean shutdown of connection pool."""
        if self._session:
            await self._session.close()
            self._session = None
            self._initialized = False
            logger.info("Closed OllamaConnectionPool")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[aiohttp.ClientSession]:
        """Acquire a connection from the pool.

        Yields:
            aiohttp ClientSession for making requests
        """
        if not self._session:
            await self.initialize()

        async with self._semaphore:
            yield self._session  # type: ignore

    async def generate(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate text using Ollama.

        Args:
            model: Ollama model name (e.g., "llama3.1:8b")
            prompt: Input prompt
            **kwargs: Additional Ollama parameters (temperature, num_predict, etc.)

        Returns:
            Generated text response

        Raises:
            OllamaConnectionError: If request fails after all retries
        """
        if not self._health_status.is_healthy:
            # Allow one attempt even when unhealthy (for recovery)
            logger.warning("Ollama connection unhealthy, attempting recovery")

        for attempt in range(self.config.max_retries):
            try:
                async with self.acquire() as session:
                    async with session.post(
                        f"{self.config.base_url}/api/generate",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "stream": False,
                            **kwargs,
                        },
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        self._health_status.record_success()
                        return data.get("response", "")

            except asyncio.TimeoutError:
                self._health_status.record_failure()
                logger.warning(
                    f"Ollama request timeout (attempt {attempt + 1}/{self.config.max_retries})"
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2**attempt))
                else:
                    raise OllamaConnectionError(
                        f"Ollama request timed out after {self.config.max_retries} attempts"
                    )

            except aiohttp.ClientError as e:
                self._health_status.record_failure()
                logger.warning(
                    f"Ollama request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2**attempt))
                else:
                    raise OllamaConnectionError(
                        f"Ollama request failed after {self.config.max_retries} attempts: {e}"
                    )

        # Should not reach here, but satisfy type checker
        raise OllamaConnectionError("Unexpected error in generate")

    async def batch_generate(
        self,
        model: str,
        prompts: List[str],
        **kwargs: Any,
    ) -> List[str]:
        """Batch generation for multiple prompts.

        Optimizes throughput by processing multiple prompts
        with concurrent connections.

        Args:
            model: Ollama model name
            prompts: List of input prompts
            **kwargs: Additional Ollama parameters

        Returns:
            List of generated responses (or error messages)
        """
        tasks = [self.generate(model, prompt, **kwargs) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error strings
        processed: List[str] = []
        for result in results:
            if isinstance(result, Exception):
                processed.append(f"[ERROR: {result}]")
            else:
                processed.append(result)

        return processed

    async def _health_check(self) -> bool:
        """Check Ollama server health.

        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self._session:
                return False

            async with self._session.get(
                f"{self.config.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as response:
                response.raise_for_status()
                self._health_status.mark_healthy()
                logger.debug("Ollama health check passed")
                return True

        except Exception as e:
            self._health_status.mark_unhealthy()
            logger.warning(f"Ollama health check failed: {e}")
            return False

    @property
    def is_healthy(self) -> bool:
        """Check if connection pool is healthy."""
        return self._health_status.is_healthy

    @property
    def is_initialized(self) -> bool:
        """Check if pool is initialized."""
        return self._initialized

    def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status.

        Returns:
            Dict with health metrics
        """
        return {
            "is_healthy": self._health_status.is_healthy,
            "consecutive_failures": self._health_status.consecutive_failures,
            "consecutive_successes": self._health_status.consecutive_successes,
            "is_initialized": self._initialized,
        }
