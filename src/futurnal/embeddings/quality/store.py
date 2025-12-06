"""SQLite-backed Quality Metrics Store.

Provides persistent storage for embedding quality metrics enabling:
- Trend analysis over time
- Low-quality embedding identification
- Historical quality queries

Follows existing SQLite patterns from orchestrator JobQueue.

Production Plan Reference:
docs/phase-1/vector-embedding-service-production-plan/05-quality-evolution.md
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterator, List, Optional

from futurnal.embeddings.quality.metrics import EmbeddingQualityMetrics

logger = logging.getLogger(__name__)


class QualityMetricsStore:
    """SQLite-backed storage for embedding quality metrics.

    Thread-safe store for quality metrics persistence enabling:
    - Trend analysis over time
    - Low-quality embedding identification
    - Historical quality queries for evolution tracking

    Example:
        store = QualityMetricsStore(db_path=Path("~/.futurnal/quality_metrics.db"))

        # Insert metrics
        store.insert(metrics)

        # Query low-quality embeddings
        low_quality_ids = store.query_low_quality(min_quality_score=0.6, limit=100)

        # Get entity history for trend analysis
        history = store.get_entity_history(entity_id="ent_123", lookback_days=30)
    """

    # Database schema version for migrations
    SCHEMA_VERSION = 1

    def __init__(
        self,
        db_path: Optional[Path] = None,
        in_memory: bool = False,
    ) -> None:
        """Initialize the quality metrics store.

        Args:
            db_path: Path to SQLite database file. If None, uses default location.
            in_memory: If True, use in-memory database (for testing).
        """
        self._in_memory = in_memory
        self._persistent_conn: Optional[sqlite3.Connection] = None

        if in_memory:
            self._db_path = ":memory:"
            # Create persistent connection for in-memory database
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
        else:
            if db_path is None:
                db_path = Path.home() / ".futurnal" / "quality_metrics.db"
            db_path = Path(db_path).expanduser()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db_path = str(db_path)

        self._lock = Lock()
        self._init_schema()

        logger.info(f"Initialized QualityMetricsStore at {self._db_path}")

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection with row factory.

        For in-memory databases, uses persistent connection to maintain schema.
        For file-based databases, creates new connection per operation.

        Yields:
            SQLite connection configured with row factory.
        """
        if self._persistent_conn is not None:
            # Use persistent connection for in-memory database
            yield self._persistent_conn
        else:
            # Create new connection for file-based database
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()

            # Create main quality metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    embedding_id TEXT UNIQUE NOT NULL,
                    entity_id TEXT NOT NULL,
                    extraction_confidence REAL NOT NULL,
                    extraction_quality_score REAL,
                    embedding_similarity_score REAL,
                    embedding_coherence REAL,
                    embedding_distinctiveness REAL,
                    temporal_accuracy REAL,
                    causal_pattern_quality REAL,
                    embedding_latency_ms REAL NOT NULL,
                    model_id TEXT NOT NULL,
                    vector_dimension INTEGER NOT NULL,
                    quality_trend TEXT CHECK (quality_trend IN ('improving', 'stable', 'degrading', NULL)),
                    reembedding_count INTEGER DEFAULT 0,
                    last_reembedded TEXT,
                    created_at TEXT NOT NULL,
                    last_validated TEXT NOT NULL,
                    overall_quality_score REAL
                )
            """)

            # Create indexes for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_entity_id
                ON embedding_quality_metrics(entity_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_extraction_confidence
                ON embedding_quality_metrics(extraction_confidence)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_overall_quality
                ON embedding_quality_metrics(overall_quality_score)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON embedding_quality_metrics(created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_id
                ON embedding_quality_metrics(model_id)
            """)

            # Create schema version table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_info (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            # Set schema version
            cursor.execute("""
                INSERT OR REPLACE INTO schema_info (key, value)
                VALUES ('version', ?)
            """, (str(self.SCHEMA_VERSION),))

            conn.commit()

    def insert(self, metrics: EmbeddingQualityMetrics) -> None:
        """Insert quality metrics record.

        Args:
            metrics: EmbeddingQualityMetrics to insert.

        Raises:
            sqlite3.IntegrityError: If embedding_id already exists.
        """
        data = metrics.to_sqlite_dict()

        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO embedding_quality_metrics (
                    embedding_id, entity_id, extraction_confidence,
                    extraction_quality_score, embedding_similarity_score,
                    embedding_coherence, embedding_distinctiveness,
                    temporal_accuracy, causal_pattern_quality,
                    embedding_latency_ms, model_id, vector_dimension,
                    quality_trend, reembedding_count, last_reembedded,
                    created_at, last_validated, overall_quality_score
                ) VALUES (
                    :embedding_id, :entity_id, :extraction_confidence,
                    :extraction_quality_score, :embedding_similarity_score,
                    :embedding_coherence, :embedding_distinctiveness,
                    :temporal_accuracy, :causal_pattern_quality,
                    :embedding_latency_ms, :model_id, :vector_dimension,
                    :quality_trend, :reembedding_count, :last_reembedded,
                    :created_at, :last_validated, :overall_quality_score
                )
            """, data)
            conn.commit()

        logger.debug(f"Inserted quality metrics for embedding {metrics.embedding_id}")

    def update(self, embedding_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing metrics record.

        Args:
            embedding_id: ID of embedding to update.
            updates: Dictionary of field names to new values.

        Returns:
            True if a record was updated, False if not found.
        """
        if not updates:
            return False

        # Build SET clause dynamically
        set_clauses = []
        values = []
        for key, value in updates.items():
            set_clauses.append(f"{key} = ?")
            # Convert datetime to ISO string if needed
            if isinstance(value, datetime):
                value = value.isoformat()
            values.append(value)

        values.append(embedding_id)

        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                UPDATE embedding_quality_metrics
                SET {', '.join(set_clauses)}
                WHERE embedding_id = ?
                """,
                values,
            )
            conn.commit()
            updated = cursor.rowcount > 0

        if updated:
            logger.debug(f"Updated quality metrics for embedding {embedding_id}")

        return updated

    def get_by_embedding_id(
        self,
        embedding_id: str,
    ) -> Optional[EmbeddingQualityMetrics]:
        """Get metrics by embedding ID.

        Args:
            embedding_id: Unique embedding identifier.

        Returns:
            EmbeddingQualityMetrics if found, None otherwise.
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM embedding_quality_metrics
                WHERE embedding_id = ?
                """,
                (embedding_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return EmbeddingQualityMetrics.from_sqlite_row(dict(row))

    def get_by_entity_id(self, entity_id: str) -> List[EmbeddingQualityMetrics]:
        """Get all metrics for an entity.

        Returns metrics sorted by created_at ascending (oldest first).

        Args:
            entity_id: PKG entity ID.

        Returns:
            List of EmbeddingQualityMetrics for this entity.
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM embedding_quality_metrics
                WHERE entity_id = ?
                ORDER BY created_at ASC
                """,
                (entity_id,),
            )
            rows = cursor.fetchall()

        return [EmbeddingQualityMetrics.from_sqlite_row(dict(row)) for row in rows]

    def query_low_quality(
        self,
        min_quality_score: float,
        limit: int = 100,
    ) -> List[str]:
        """Query entity IDs with embeddings below quality threshold.

        Returns unique entity IDs where overall_quality_score or
        extraction_confidence is below the threshold.

        Args:
            min_quality_score: Minimum acceptable quality score.
            limit: Maximum number of entity IDs to return.

        Returns:
            List of entity IDs needing re-embedding.
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT entity_id
                FROM embedding_quality_metrics
                WHERE overall_quality_score < ?
                   OR overall_quality_score IS NULL
                   OR extraction_confidence < ?
                   OR embedding_coherence < 0.5
                ORDER BY COALESCE(overall_quality_score, 0) ASC
                LIMIT ?
                """,
                (min_quality_score, min_quality_score, limit),
            )
            rows = cursor.fetchall()

        return [row["entity_id"] for row in rows]

    def query_by_timerange(
        self,
        entity_id: str,
        start: datetime,
        end: datetime,
    ) -> List[EmbeddingQualityMetrics]:
        """Query metrics within time range for trend analysis.

        Args:
            entity_id: PKG entity ID.
            start: Start of time range (inclusive).
            end: End of time range (inclusive).

        Returns:
            List of metrics within the time range, sorted by created_at.
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM embedding_quality_metrics
                WHERE entity_id = ?
                  AND created_at >= ?
                  AND created_at <= ?
                ORDER BY created_at ASC
                """,
                (entity_id, start.isoformat(), end.isoformat()),
            )
            rows = cursor.fetchall()

        return [EmbeddingQualityMetrics.from_sqlite_row(dict(row)) for row in rows]

    def get_entity_history(
        self,
        entity_id: str,
        lookback_days: int = 30,
    ) -> List[EmbeddingQualityMetrics]:
        """Get historical metrics for trend calculation.

        Args:
            entity_id: PKG entity ID.
            lookback_days: Number of days to look back.

        Returns:
            List of metrics within lookback period, sorted by created_at.
        """
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)

        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM embedding_quality_metrics
                WHERE entity_id = ?
                  AND created_at > ?
                ORDER BY created_at ASC
                """,
                (entity_id, cutoff.isoformat()),
            )
            rows = cursor.fetchall()

        return [EmbeddingQualityMetrics.from_sqlite_row(dict(row)) for row in rows]

    def delete_by_embedding_id(self, embedding_id: str) -> bool:
        """Delete metrics by embedding ID.

        Args:
            embedding_id: Unique embedding identifier.

        Returns:
            True if a record was deleted, False otherwise.
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM embedding_quality_metrics
                WHERE embedding_id = ?
                """,
                (embedding_id,),
            )
            conn.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            logger.debug(f"Deleted quality metrics for embedding {embedding_id}")

        return deleted

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics for the quality metrics store.

        Returns:
            Dictionary with statistics including total count, averages, etc.
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()

            # Total count
            cursor.execute("SELECT COUNT(*) FROM embedding_quality_metrics")
            total_count = cursor.fetchone()[0]

            # Average quality scores
            cursor.execute("""
                SELECT
                    AVG(overall_quality_score) as avg_quality,
                    AVG(extraction_confidence) as avg_confidence,
                    AVG(embedding_coherence) as avg_coherence,
                    AVG(embedding_latency_ms) as avg_latency
                FROM embedding_quality_metrics
            """)
            row = cursor.fetchone()

            # Trend distribution
            cursor.execute("""
                SELECT quality_trend, COUNT(*) as count
                FROM embedding_quality_metrics
                WHERE quality_trend IS NOT NULL
                GROUP BY quality_trend
            """)
            trend_rows = cursor.fetchall()
            trend_distribution = {row["quality_trend"]: row["count"] for row in trend_rows}

            # Low quality count
            cursor.execute("""
                SELECT COUNT(DISTINCT entity_id) as count
                FROM embedding_quality_metrics
                WHERE overall_quality_score < 0.6
                   OR overall_quality_score IS NULL
            """)
            low_quality_count = cursor.fetchone()[0]

        return {
            "total_count": total_count,
            "avg_quality_score": row["avg_quality"] if row else None,
            "avg_extraction_confidence": row["avg_confidence"] if row else None,
            "avg_embedding_coherence": row["avg_coherence"] if row else None,
            "avg_latency_ms": row["avg_latency"] if row else None,
            "trend_distribution": trend_distribution,
            "low_quality_entity_count": low_quality_count,
        }

    def close(self) -> None:
        """Close any resources.

        For in-memory databases, closes the persistent connection.
        For file-based databases, connections are per-operation so this is a no-op.
        """
        if self._persistent_conn is not None:
            self._persistent_conn.close()
            self._persistent_conn = None
            logger.debug("Closed persistent in-memory connection")
