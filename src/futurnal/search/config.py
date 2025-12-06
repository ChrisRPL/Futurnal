"""Search Configuration Module.

Defines configuration for the Hybrid Search API:
- TemporalSearchConfig: Temporal query engine settings
- SearchConfig: Top-level search configuration

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/

Option B Compliance:
- Local-first: All processing on-device
- Configurable quality thresholds for experiential learning feedback
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from futurnal.search.hybrid.config import HybridSearchConfig


@dataclass
class TemporalSearchConfig:
    """Configuration for temporal search engine.

    All settings have sensible defaults aligned with production plan targets.

    Example:
        >>> config = TemporalSearchConfig(
        ...     decay_half_life_days=14.0,  # More aggressive recency weighting
        ...     min_pattern_occurrences=5,   # Stricter pattern significance
        ... )
    """

    # Decay scoring configuration
    decay_half_life_days: float = 30.0
    """Half-life in days for exponential decay scoring. Default: 30 days."""

    enable_decay_by_default: bool = True
    """Apply decay scoring by default. Default: True."""

    # Pattern matching configuration
    default_max_gap_days: int = 30
    """Default maximum gap between events in sequences. Default: 30 days."""

    min_pattern_occurrences: int = 3
    """Minimum occurrences for pattern significance. Default: 3."""

    min_pattern_length: int = 2
    """Minimum pattern length for recurring pattern detection. Default: 2."""

    max_pattern_length: int = 5
    """Maximum pattern length to consider. Default: 5."""

    # Correlation detection configuration
    correlation_min_occurrences: int = 3
    """Minimum co-occurrences for correlation significance. Default: 3."""

    correlation_significance_threshold: float = 0.5
    """Minimum correlation strength to report. Default: 0.5."""

    # Query limits and performance
    default_query_limit: int = 100
    """Default number of results per query. Default: 100."""

    max_query_limit: int = 1000
    """Maximum allowed results per query. Default: 1000."""

    query_timeout_ms: float = 5000.0
    """Query timeout in milliseconds. Default: 5000ms (5s)."""

    # Hybrid search weights
    vector_similarity_weight: float = 0.3
    """Weight for vector similarity in hybrid results. Default: 0.3."""

    graph_relevance_weight: float = 0.7
    """Weight for graph relevance in hybrid results. Default: 0.7."""

    # Performance targets (for monitoring)
    target_latency_ms: float = 1000.0
    """Target query latency in milliseconds. Default: 1000ms."""

    cache_ttl_seconds: int = 300
    """Cache TTL in seconds. Default: 300s (5 min)."""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.decay_half_life_days <= 0:
            raise ValueError("decay_half_life_days must be positive")

        if self.vector_similarity_weight + self.graph_relevance_weight != 1.0:
            # Normalize weights
            total = self.vector_similarity_weight + self.graph_relevance_weight
            self.vector_similarity_weight /= total
            self.graph_relevance_weight /= total

        if self.min_pattern_length > self.max_pattern_length:
            raise ValueError(
                "min_pattern_length cannot exceed max_pattern_length"
            )


@dataclass
class CausalSearchConfig:
    """Configuration for causal chain retrieval.

    All settings have sensible defaults aligned with production plan targets.

    Production Plan Reference:
    docs/phase-1/hybrid-search-api-production-plan/02-causal-chain-retrieval.md

    Example:
        >>> config = CausalSearchConfig(
        ...     default_max_hops=5,
        ...     default_min_confidence=0.7,
        ... )
    """

    # Traversal configuration
    default_max_hops: int = 3
    """Default maximum hops for causal traversal. Default: 3."""

    max_allowed_hops: int = 10
    """Maximum allowed hops (cannot be exceeded). Default: 10."""

    default_min_confidence: float = 0.6
    """Default minimum causal confidence. Default: 0.6."""

    # Correlation detection
    default_correlation_strength: float = 0.5
    """Default minimum correlation strength. Default: 0.5."""

    # Query limits and performance
    default_result_limit: int = 20
    """Default number of results per query. Default: 20."""

    max_result_limit: int = 100
    """Maximum allowed results per query. Default: 100."""

    query_timeout_ms: float = 2000.0
    """Query timeout in milliseconds. Target: <2s. Default: 2000ms."""

    # Performance targets (for monitoring)
    target_latency_ms: float = 2000.0
    """Target query latency in milliseconds. Default: 2000ms (2s)."""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.default_max_hops > self.max_allowed_hops:
            raise ValueError(
                "default_max_hops cannot exceed max_allowed_hops"
            )

        if not 0.0 <= self.default_min_confidence <= 1.0:
            raise ValueError(
                "default_min_confidence must be between 0.0 and 1.0"
            )

        if not 0.0 <= self.default_correlation_strength <= 1.0:
            raise ValueError(
                "default_correlation_strength must be between 0.0 and 1.0"
            )


@dataclass
class SearchConfig:
    """Top-level search configuration.

    Aggregates all search subsystem configurations.

    Example:
        >>> config = SearchConfig(
        ...     temporal=TemporalSearchConfig(decay_half_life_days=14.0),
        ... )
        >>> engine = TemporalQueryEngine(..., config=config.temporal)
    """

    temporal: TemporalSearchConfig = field(default_factory=TemporalSearchConfig)
    """Temporal search configuration."""

    causal: CausalSearchConfig = field(default_factory=CausalSearchConfig)
    """Causal chain retrieval configuration."""

    # Hybrid search config is lazily imported to avoid circular dependencies
    _hybrid: Optional["HybridSearchConfig"] = field(default=None, repr=False)
    """Schema-aware hybrid retrieval configuration (Module 03)."""

    # Audit logging
    enable_audit_logging: bool = True
    """Enable audit logging for search queries. Default: True."""

    audit_log_content: bool = False
    """Log query content (privacy-sensitive). Default: False."""

    @property
    def hybrid(self) -> "HybridSearchConfig":
        """Get hybrid search configuration (lazy loading).

        Returns:
            HybridSearchConfig instance
        """
        if self._hybrid is None:
            from futurnal.search.hybrid.config import HybridSearchConfig
            self._hybrid = HybridSearchConfig()
        return self._hybrid

    @hybrid.setter
    def hybrid(self, value: "HybridSearchConfig") -> None:
        """Set hybrid search configuration."""
        self._hybrid = value

    @classmethod
    def from_env(cls) -> "SearchConfig":
        """Load configuration from environment variables.

        Environment variables:
        - FUTURNAL_SEARCH_DECAY_HALF_LIFE: Decay half-life in days
        - FUTURNAL_SEARCH_VECTOR_WEIGHT: Vector similarity weight
        - FUTURNAL_SEARCH_ENABLE_AUDIT: Enable audit logging (true/false)

        Returns:
            SearchConfig with environment overrides applied.
        """
        import os

        temporal_config = TemporalSearchConfig()

        # Override from environment
        if half_life := os.getenv("FUTURNAL_SEARCH_DECAY_HALF_LIFE"):
            temporal_config.decay_half_life_days = float(half_life)

        if vector_weight := os.getenv("FUTURNAL_SEARCH_VECTOR_WEIGHT"):
            temporal_config.vector_similarity_weight = float(vector_weight)
            temporal_config.graph_relevance_weight = 1.0 - float(vector_weight)

        enable_audit = os.getenv("FUTURNAL_SEARCH_ENABLE_AUDIT", "true")

        return cls(
            temporal=temporal_config,
            enable_audit_logging=enable_audit.lower() == "true",
        )
