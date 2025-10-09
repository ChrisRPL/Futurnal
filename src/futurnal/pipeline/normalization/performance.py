"""Performance monitoring for normalization pipeline.

This module provides throughput tracking and performance metrics for the document
normalization pipeline. Monitors processing rates (MB/s), document counts, and
per-format statistics to validate production performance targets.

Key Features:
- Real-time throughput calculation (MB/s)
- Per-format performance statistics
- Document and byte counters
- Processing duration tracking
- Telemetry-compatible metrics export
- Privacy-preserving (metadata only, no content)

Design Philosophy:
- Throughput target: ≥5 MB/s for mixed document types
- Per-format breakdown for optimization insights
- Lightweight tracking with minimal overhead
- Integration with existing telemetry infrastructure
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

from ..models import DocumentFormat

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Performance Statistics Models
# ---------------------------------------------------------------------------


@dataclass
class FormatStats:
    """Performance statistics for a specific document format.

    Tracks processing metrics per format to identify optimization opportunities
    and validate format-specific performance targets.

    Attributes:
        format: Document format type
        document_count: Number of documents processed
        total_bytes: Total bytes processed for this format
        total_duration_ms: Total processing time in milliseconds
        avg_throughput_mbps: Average throughput in MB/s
        last_updated: Timestamp of last update
    """

    format: DocumentFormat
    document_count: int = 0
    total_bytes: int = 0
    total_duration_ms: float = 0.0
    avg_throughput_mbps: float = 0.0
    last_updated: Optional[datetime] = None

    def update(self, size_bytes: int, duration_ms: float) -> None:
        """Update statistics with new document metrics.

        Args:
            size_bytes: Document size in bytes
            duration_ms: Processing duration in milliseconds
        """
        self.document_count += 1
        self.total_bytes += size_bytes
        self.total_duration_ms += duration_ms

        # Recalculate average throughput
        if self.total_duration_ms > 0:
            duration_seconds = self.total_duration_ms / 1000.0
            mb_processed = self.total_bytes / (1024 * 1024)
            self.avg_throughput_mbps = mb_processed / duration_seconds
        else:
            self.avg_throughput_mbps = 0.0

        self.last_updated = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        """Export statistics as dictionary.

        Returns:
            Dictionary with format statistics
        """
        return {
            "format": self.format.value,
            "document_count": self.document_count,
            "total_bytes": self.total_bytes,
            "total_mb": round(self.total_bytes / (1024 * 1024), 2),
            "total_duration_ms": round(self.total_duration_ms, 2),
            "avg_throughput_mbps": round(self.avg_throughput_mbps, 2),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


@dataclass
class PerformanceSnapshot:
    """Point-in-time snapshot of performance metrics.

    Captures current performance state for reporting and analysis.

    Attributes:
        timestamp: Snapshot timestamp
        total_documents: Total documents processed
        total_bytes: Total bytes processed
        total_duration_ms: Total processing time
        current_throughput_mbps: Current throughput in MB/s
        documents_per_second: Document processing rate
        format_breakdown: Per-format statistics
    """

    timestamp: datetime
    total_documents: int
    total_bytes: int
    total_duration_ms: float
    current_throughput_mbps: float
    documents_per_second: float
    format_breakdown: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Export snapshot as dictionary.

        Returns:
            Dictionary with performance snapshot
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_documents": self.total_documents,
            "total_bytes": self.total_bytes,
            "total_mb": round(self.total_bytes / (1024 * 1024), 2),
            "total_duration_ms": round(self.total_duration_ms, 2),
            "total_duration_seconds": round(self.total_duration_ms / 1000.0, 2),
            "current_throughput_mbps": round(self.current_throughput_mbps, 2),
            "documents_per_second": round(self.documents_per_second, 2),
            "format_breakdown": self.format_breakdown,
        }


# ---------------------------------------------------------------------------
# Performance Monitor
# ---------------------------------------------------------------------------


class PerformanceMonitor:
    """Monitor normalization pipeline performance and throughput.

    Tracks document processing rates, throughput (MB/s), and per-format statistics
    to validate production performance targets and identify optimization opportunities.

    Example:
        >>> monitor = PerformanceMonitor()
        >>> monitor.start()
        >>> monitor.record_document(size_bytes=1024000, format=DocumentFormat.PDF, duration_ms=250.0)
        >>> throughput = monitor.get_throughput_mbps()
        >>> print(f"Current throughput: {throughput:.2f} MB/s")
    """

    def __init__(self):
        """Initialize performance monitor."""
        # Session tracking
        self.start_time: Optional[float] = None
        self.session_start: Optional[datetime] = None

        # Counters
        self.bytes_processed: int = 0
        self.documents_processed: int = 0

        # Timing
        self.total_processing_time_ms: float = 0.0

        # Per-format statistics
        self._format_stats: Dict[DocumentFormat, FormatStats] = {}

        # Metrics
        self.last_snapshot: Optional[PerformanceSnapshot] = None

        logger.debug("PerformanceMonitor initialized")

    def start(self) -> None:
        """Start performance tracking session.

        Initializes timing counters for throughput calculation.
        Can be called multiple times to reset tracking.
        """
        self.start_time = time.time()
        self.session_start = datetime.now(timezone.utc)

        logger.info("Performance monitoring session started")

    def record_document(
        self, *, size_bytes: int, format: DocumentFormat, duration_ms: float
    ) -> None:
        """Record a processed document for performance tracking.

        Args:
            size_bytes: Document size in bytes
            format: Document format type
            duration_ms: Processing duration in milliseconds
        """
        # Update overall counters
        self.bytes_processed += size_bytes
        self.documents_processed += 1
        self.total_processing_time_ms += duration_ms

        # Update format-specific statistics
        if format not in self._format_stats:
            self._format_stats[format] = FormatStats(format=format)

        self._format_stats[format].update(size_bytes, duration_ms)

        # Log milestone progress
        if self.documents_processed % 100 == 0:
            current_throughput = self.get_throughput_mbps()
            logger.info(
                f"Processed {self.documents_processed} documents, "
                f"current throughput: {current_throughput:.2f} MB/s"
            )

    def get_throughput_mbps(self) -> float:
        """Calculate current throughput in MB/s.

        Uses elapsed wall-clock time since session start for accurate
        real-time throughput measurement.

        Returns:
            Current throughput in megabytes per second
        """
        if not self.start_time:
            return 0.0

        elapsed_seconds = time.time() - self.start_time
        if elapsed_seconds == 0:
            return 0.0

        mb_processed = self.bytes_processed / (1024 * 1024)
        return mb_processed / elapsed_seconds

    def get_documents_per_second(self) -> float:
        """Calculate document processing rate.

        Returns:
            Documents processed per second
        """
        if not self.start_time:
            return 0.0

        elapsed_seconds = time.time() - self.start_time
        if elapsed_seconds == 0:
            return 0.0

        return self.documents_processed / elapsed_seconds

    def get_format_stats(self, format: DocumentFormat) -> Optional[FormatStats]:
        """Get performance statistics for specific format.

        Args:
            format: Document format

        Returns:
            FormatStats if format has been processed, None otherwise
        """
        return self._format_stats.get(format)

    def get_all_format_stats(self) -> Dict[DocumentFormat, FormatStats]:
        """Get performance statistics for all formats.

        Returns:
            Dictionary mapping formats to their statistics
        """
        return self._format_stats.copy()

    def get_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot.

        Returns:
            PerformanceSnapshot with current metrics
        """
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_documents=self.documents_processed,
            total_bytes=self.bytes_processed,
            total_duration_ms=self.total_processing_time_ms,
            current_throughput_mbps=self.get_throughput_mbps(),
            documents_per_second=self.get_documents_per_second(),
            format_breakdown={
                format.value: stats.to_dict()
                for format, stats in self._format_stats.items()
            },
        )

        self.last_snapshot = snapshot
        return snapshot

    def get_metrics(self) -> dict:
        """Get performance metrics for telemetry export.

        Returns:
            Dictionary with comprehensive performance metrics
        """
        throughput_mbps = self.get_throughput_mbps()
        docs_per_sec = self.get_documents_per_second()

        metrics = {
            # Overall metrics
            "documents_processed": self.documents_processed,
            "bytes_processed": self.bytes_processed,
            "mb_processed": round(self.bytes_processed / (1024 * 1024), 2),
            "total_processing_time_ms": round(self.total_processing_time_ms, 2),

            # Rates
            "throughput_mbps": round(throughput_mbps, 2),
            "documents_per_second": round(docs_per_sec, 2),

            # Efficiency
            "avg_processing_time_ms": (
                round(self.total_processing_time_ms / self.documents_processed, 2)
                if self.documents_processed > 0
                else 0.0
            ),
            "avg_document_size_mb": (
                round(self.bytes_processed / self.documents_processed / (1024 * 1024), 2)
                if self.documents_processed > 0
                else 0.0
            ),

            # Session info
            "session_start": self.session_start.isoformat() if self.session_start else None,
            "session_duration_seconds": (
                round(time.time() - self.start_time, 2) if self.start_time else 0.0
            ),

            # Performance validation
            "meets_throughput_target": throughput_mbps >= 5.0,
            "throughput_target_mbps": 5.0,

            # Per-format breakdown
            "format_stats": {
                format.value: stats.to_dict()
                for format, stats in self._format_stats.items()
            },
        }

        return metrics

    def reset(self) -> None:
        """Reset all performance metrics and counters.

        Useful for starting a new measurement session.
        """
        self.start_time = None
        self.session_start = None
        self.bytes_processed = 0
        self.documents_processed = 0
        self.total_processing_time_ms = 0.0
        self._format_stats.clear()
        self.last_snapshot = None

        logger.info("Performance monitor reset")

    def log_summary(self) -> None:
        """Log performance summary to logger.

        Useful for end-of-session reporting.
        """
        metrics = self.get_metrics()

        logger.info("=" * 60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Documents Processed: {metrics['documents_processed']}")
        logger.info(f"Data Processed: {metrics['mb_processed']} MB")
        logger.info(f"Session Duration: {metrics['session_duration_seconds']}s")
        logger.info(f"Throughput: {metrics['throughput_mbps']} MB/s")
        logger.info(f"Processing Rate: {metrics['documents_per_second']} docs/s")
        logger.info(
            f"Target Met: {'✓' if metrics['meets_throughput_target'] else '✗'} "
            f"(target: {metrics['throughput_target_mbps']} MB/s)"
        )

        if self._format_stats:
            logger.info("\nPer-Format Statistics:")
            for format, stats in self._format_stats.items():
                logger.info(
                    f"  {format.value}: {stats.document_count} docs, "
                    f"{stats.avg_throughput_mbps:.2f} MB/s"
                )

        logger.info("=" * 60)
