"""Hybrid Search Configuration.

Configuration for schema-aware hybrid retrieval:
- Weight defaults for vector/graph fusion
- Entity type-specific strategy weights
- Schema compatibility thresholds
- Performance targets

Production Plan Reference:
docs/phase-1/hybrid-search-api-production-plan/03-schema-aware-retrieval.md

Option B Compliance:
- Local-first processing targets
- Configurable quality thresholds
- Schema drift handling parameters
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class HybridSearchConfig:
    """Configuration for schema-aware hybrid retrieval.

    All settings have sensible defaults aligned with production plan targets.

    Example:
        >>> config = HybridSearchConfig(
        ...     default_vector_weight=0.6,
        ...     target_latency_ms=800.0,
        ... )
        >>> retrieval = SchemaAwareRetrieval(..., config=config)

    Attributes:
        default_vector_weight: Default weight for vector similarity
        default_graph_weight: Default weight for graph traversal
        event_vector_weight: Vector weight for Event entity queries
        event_graph_weight: Graph weight for Event entity queries
        code_vector_weight: Vector weight for CodeEntity queries
        code_graph_weight: Graph weight for CodeEntity queries
        document_vector_weight: Vector weight for Document queries
        document_graph_weight: Graph weight for Document queries
        schema_drift_threshold: Maximum version difference before re-embedding
        minor_drift_penalty: Score multiplier for 1 version difference
        moderate_drift_penalty: Score multiplier for 2-3 version difference
        target_latency_ms: Target latency in milliseconds
        vector_top_k_multiplier: Multiplier for initial vector retrieval
        graph_expansion_limit: Maximum results from graph expansion
        code_keywords: Keywords triggering CODE embedding type
        document_keywords: Keywords triggering DOCUMENT embedding type
        temporal_keywords: Keywords enhancing TEMPORAL detection
    """

    # Weight defaults
    default_vector_weight: float = 0.5
    """Default weight for vector similarity in hybrid fusion."""

    default_graph_weight: float = 0.5
    """Default weight for graph traversal in hybrid fusion."""

    # Entity type strategy weights (from production plan)
    event_vector_weight: float = 0.4
    """Vector weight for Event entities (lower - temporal relationships matter)."""

    event_graph_weight: float = 0.6
    """Graph weight for Event entities (higher - temporal relationships matter)."""

    code_vector_weight: float = 0.7
    """Vector weight for CodeEntity (higher - semantic similarity matters)."""

    code_graph_weight: float = 0.3
    """Graph weight for CodeEntity (lower - semantic similarity matters)."""

    document_vector_weight: float = 0.6
    """Vector weight for Document entities (higher - content focus)."""

    document_graph_weight: float = 0.4
    """Graph weight for Document entities (lower - content focus)."""

    # Schema compatibility thresholds (from production plan)
    schema_drift_threshold: int = 3
    """Maximum version diff before requiring re-embedding."""

    minor_drift_penalty: float = 0.95
    """Score multiplier for 1 version difference (5% penalty)."""

    moderate_drift_penalty: float = 0.85
    """Score multiplier for 2-3 version difference (15% penalty)."""

    # Performance targets
    target_latency_ms: float = 1000.0
    """Target query latency in milliseconds (production plan: <1s)."""

    vector_top_k_multiplier: int = 2
    """Multiplier for initial vector retrieval (retrieve 2x for fusion)."""

    graph_expansion_limit: int = 50
    """Maximum results from graph expansion per intent type."""

    max_seed_entities: int = 10
    """Maximum seed entities for graph expansion."""

    # Query routing keywords
    code_keywords: List[str] = field(
        default_factory=lambda: [
            "code",
            "function",
            "class",
            "method",
            "implement",
            "bug",
            "error",
            "exception",
            "syntax",
            "variable",
            "parameter",
            "return",
            "import",
            "module",
            "package",
        ]
    )
    """Keywords triggering CODE embedding type selection."""

    document_keywords: List[str] = field(
        default_factory=lambda: [
            "document",
            "file",
            "note",
            "article",
            "paper",
            "report",
            "memo",
            "transcript",
            "summary",
            "abstract",
        ]
    )
    """Keywords triggering DOCUMENT embedding type selection."""

    temporal_keywords: List[str] = field(
        default_factory=lambda: [
            "when",
            "before",
            "after",
            "during",
            "yesterday",
            "today",
            "tomorrow",
            "last week",
            "next month",
            "recently",
            "timeline",
            "schedule",
            "date",
            "time",
        ]
    )
    """Keywords enhancing TEMPORAL embedding detection."""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Validate weight ranges
        if not 0.0 <= self.default_vector_weight <= 1.0:
            raise ValueError("default_vector_weight must be between 0.0 and 1.0")
        if not 0.0 <= self.default_graph_weight <= 1.0:
            raise ValueError("default_graph_weight must be between 0.0 and 1.0")

        # Normalize default weights if needed
        total = self.default_vector_weight + self.default_graph_weight
        if total > 0 and abs(total - 1.0) > 0.001:
            self.default_vector_weight /= total
            self.default_graph_weight /= total

        # Validate schema drift threshold
        if self.schema_drift_threshold < 1:
            raise ValueError("schema_drift_threshold must be at least 1")

        # Validate drift penalties
        if not 0.0 <= self.minor_drift_penalty <= 1.0:
            raise ValueError("minor_drift_penalty must be between 0.0 and 1.0")
        if not 0.0 <= self.moderate_drift_penalty <= 1.0:
            raise ValueError("moderate_drift_penalty must be between 0.0 and 1.0")

        # Validate performance targets
        if self.target_latency_ms <= 0:
            raise ValueError("target_latency_ms must be positive")
        if self.vector_top_k_multiplier < 1:
            raise ValueError("vector_top_k_multiplier must be at least 1")
        if self.graph_expansion_limit < 1:
            raise ValueError("graph_expansion_limit must be at least 1")

    def get_entity_weights(self, entity_type: str) -> Dict[str, float]:
        """Get vector/graph weights for an entity type.

        Args:
            entity_type: Entity type (Event, CodeEntity, Document, etc.)

        Returns:
            Dict with "vector" and "graph" weight keys
        """
        weights = {
            "Event": {
                "vector": self.event_vector_weight,
                "graph": self.event_graph_weight,
            },
            "CodeEntity": {
                "vector": self.code_vector_weight,
                "graph": self.code_graph_weight,
            },
            "Document": {
                "vector": self.document_vector_weight,
                "graph": self.document_graph_weight,
            },
        }

        return weights.get(
            entity_type,
            {
                "vector": self.default_vector_weight,
                "graph": self.default_graph_weight,
            },
        )

    @classmethod
    def from_env(cls) -> "HybridSearchConfig":
        """Load configuration from environment variables.

        Environment variables:
        - FUTURNAL_HYBRID_VECTOR_WEIGHT: Default vector weight
        - FUTURNAL_HYBRID_GRAPH_WEIGHT: Default graph weight
        - FUTURNAL_SCHEMA_DRIFT_THRESHOLD: Max version diff
        - FUTURNAL_HYBRID_TARGET_LATENCY_MS: Target latency

        Returns:
            HybridSearchConfig with environment overrides applied
        """
        config = cls()

        if vector_weight := os.getenv("FUTURNAL_HYBRID_VECTOR_WEIGHT"):
            config.default_vector_weight = float(vector_weight)

        if graph_weight := os.getenv("FUTURNAL_HYBRID_GRAPH_WEIGHT"):
            config.default_graph_weight = float(graph_weight)

        if drift_threshold := os.getenv("FUTURNAL_SCHEMA_DRIFT_THRESHOLD"):
            config.schema_drift_threshold = int(drift_threshold)

        if latency := os.getenv("FUTURNAL_HYBRID_TARGET_LATENCY_MS"):
            config.target_latency_ms = float(latency)

        # Re-run validation
        config.__post_init__()

        return config
