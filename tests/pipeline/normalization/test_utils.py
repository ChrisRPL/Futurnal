"""Test utilities for quality gates validation.

Provides helper functions, custom assertions, and report generators for
comprehensive quality gates testing including:
- Format coverage reporting
- Performance benchmark analysis
- Production readiness checklist validation
- Custom normalization assertions
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from futurnal.pipeline.models import DocumentFormat, NormalizedDocument


# ---------------------------------------------------------------------------
# Custom Assertions
# ---------------------------------------------------------------------------


def assert_deterministic_hash(content: str, expected_hash: Optional[str] = None) -> str:
    """Assert content produces deterministic SHA-256 hash.

    Args:
        content: Content to hash
        expected_hash: Optional expected hash to compare against

    Returns:
        Computed SHA-256 hash

    Raises:
        AssertionError: If hash doesn't match expected
    """
    computed_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    if expected_hash:
        assert computed_hash == expected_hash, (
            f"Hash mismatch: expected {expected_hash}, got {computed_hash}"
        )

    return computed_hash


def assert_documents_identical(
    doc1: NormalizedDocument,
    doc2: NormalizedDocument,
    exclude_timestamps: bool = True
) -> None:
    """Assert two normalized documents are byte-identical.

    Args:
        doc1: First document
        doc2: Second document
        exclude_timestamps: Whether to exclude timestamp fields from comparison

    Raises:
        AssertionError: If documents are not identical
    """
    # SHA-256 must be identical
    assert doc1.sha256 == doc2.sha256, (
        f"SHA-256 mismatch: {doc1.sha256} != {doc2.sha256}"
    )

    # Content must be identical
    assert doc1.content == doc2.content, "Content mismatch"

    # Chunk count must match
    assert len(doc1.chunks) == len(doc2.chunks), (
        f"Chunk count mismatch: {len(doc1.chunks)} != {len(doc2.chunks)}"
    )

    # Chunk content must be identical
    for idx, (chunk1, chunk2) in enumerate(zip(doc1.chunks, doc2.chunks)):
        assert chunk1.content == chunk2.content, f"Chunk {idx} content mismatch"
        assert chunk1.content_hash == chunk2.content_hash, f"Chunk {idx} hash mismatch"

    # Metadata must match (excluding timestamps if requested)
    if exclude_timestamps:
        assert_metadata_equal_excluding_timestamps(doc1.metadata, doc2.metadata)
    else:
        assert doc1.metadata == doc2.metadata, "Metadata mismatch"


def assert_metadata_equal_excluding_timestamps(meta1, meta2) -> None:
    """Assert metadata is equal excluding timestamp fields.

    Args:
        meta1: First metadata object
        meta2: Second metadata object

    Raises:
        AssertionError: If metadata differs (excluding timestamps)
    """
    # Fields to exclude from comparison
    exclude_fields = {
        "ingested_at",
        "normalized_at",
        "processing_duration_ms",
        "created_at",
        "modified_at",
        "source_id"  # source_id can differ between test runs
    }

    # Convert to dicts and remove excluded fields
    dict1 = meta1.model_dump()
    dict2 = meta2.model_dump()

    for field in exclude_fields:
        dict1.pop(field, None)
        dict2.pop(field, None)

    assert dict1 == dict2, f"Metadata mismatch: {dict1} != {dict2}"


def assert_no_content_in_string(content: str, search_string: str) -> None:
    """Assert that content does not appear in search string (for privacy tests).

    Args:
        content: Content that should NOT appear
        search_string: String to search in (e.g., log output)

    Raises:
        AssertionError: If content is found
    """
    assert content not in search_string, (
        f"Content leak detected: '{content[:50]}...' found in output"
    )


def assert_valid_sha256(hash_string: str) -> None:
    """Assert string is a valid SHA-256 hash.

    Args:
        hash_string: Hash to validate

    Raises:
        AssertionError: If not a valid SHA-256 hash
    """
    assert len(hash_string) == 64, f"Invalid hash length: {len(hash_string)}"
    assert all(c in "0123456789abcdef" for c in hash_string.lower()), (
        "Invalid hash characters"
    )


# ---------------------------------------------------------------------------
# Format Coverage Reporting
# ---------------------------------------------------------------------------


@dataclass
class FormatCoverageReport:
    """Report on format coverage testing results.

    Attributes:
        total_formats: Total number of formats tested
        successful_formats: Number of formats that passed
        failed_formats: Number of formats that failed
        format_results: Detailed results per format
        coverage_percentage: Percentage of formats covered
        timestamp: Report generation timestamp
    """

    total_formats: int
    successful_formats: int
    failed_formats: int
    format_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    coverage_percentage: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Calculate coverage percentage."""
        if self.total_formats > 0:
            self.coverage_percentage = (self.successful_formats / self.total_formats) * 100

    def add_result(
        self,
        format_name: str,
        success: bool,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a format test result.

        Args:
            format_name: Format name
            success: Whether test succeeded
            error_message: Optional error message if failed
            metadata: Optional additional metadata
        """
        self.format_results[format_name] = {
            "success": success,
            "error_message": error_message,
            "metadata": metadata or {}
        }

        if success:
            self.successful_formats += 1
        else:
            self.failed_formats += 1

        self.total_formats = len(self.format_results)
        self.__post_init__()  # Recalculate percentage

    def to_dict(self) -> dict:
        """Export report as dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_formats": self.total_formats,
                "successful_formats": self.successful_formats,
                "failed_formats": self.failed_formats,
                "coverage_percentage": round(self.coverage_percentage, 2)
            },
            "format_results": self.format_results
        }

    def save_json(self, output_path: Path) -> None:
        """Save report to JSON file.

        Args:
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> None:
        """Print human-readable summary to stdout."""
        print("\n" + "=" * 70)
        print("FORMAT COVERAGE REPORT")
        print("=" * 70)
        print(f"Total Formats Tested: {self.total_formats}")
        print(f"Successful: {self.successful_formats}")
        print(f"Failed: {self.failed_formats}")
        print(f"Coverage: {self.coverage_percentage:.2f}%")
        print("=" * 70)

        if self.format_results:
            print("\nDetailed Results:")
            for format_name, result in sorted(self.format_results.items()):
                status = "✓ PASS" if result["success"] else "✗ FAIL"
                print(f"  {format_name:15} {status}")
                if not result["success"] and result["error_message"]:
                    print(f"    Error: {result['error_message']}")

        print("=" * 70 + "\n")


def create_format_coverage_report() -> FormatCoverageReport:
    """Create a new format coverage report.

    Returns:
        Empty FormatCoverageReport instance
    """
    # Get all DocumentFormat values
    all_formats = [f.value for f in DocumentFormat]
    return FormatCoverageReport(
        total_formats=0,
        successful_formats=0,
        failed_formats=0
    )


# ---------------------------------------------------------------------------
# Performance Analysis
# ---------------------------------------------------------------------------


@dataclass
class PerformanceAnalysis:
    """Analysis of performance benchmark results.

    Attributes:
        throughput_mbps: Overall throughput in MB/s
        meets_target: Whether performance meets ≥5 MB/s target
        documents_processed: Total documents processed
        total_mb_processed: Total data processed in MB
        avg_processing_time_ms: Average processing time per document
        format_breakdown: Per-format performance statistics
        timestamp: Analysis timestamp
    """

    throughput_mbps: float
    meets_target: bool
    documents_processed: int
    total_mb_processed: float
    avg_processing_time_ms: float
    format_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_performance_metrics(cls, metrics: dict) -> PerformanceAnalysis:
        """Create analysis from performance monitor metrics.

        Args:
            metrics: Metrics dictionary from PerformanceMonitor

        Returns:
            PerformanceAnalysis instance
        """
        return cls(
            throughput_mbps=metrics.get("throughput_mbps", 0.0),
            meets_target=metrics.get("meets_throughput_target", False),
            documents_processed=metrics.get("documents_processed", 0),
            total_mb_processed=metrics.get("mb_processed", 0.0),
            avg_processing_time_ms=metrics.get("avg_processing_time_ms", 0.0),
            format_breakdown=metrics.get("format_stats", {})
        )

    def to_dict(self) -> dict:
        """Export analysis as dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_performance": {
                "throughput_mbps": round(self.throughput_mbps, 2),
                "meets_target": self.meets_target,
                "target_mbps": 5.0,
                "documents_processed": self.documents_processed,
                "total_mb_processed": round(self.total_mb_processed, 2),
                "avg_processing_time_ms": round(self.avg_processing_time_ms, 2)
            },
            "format_breakdown": self.format_breakdown
        }

    def save_json(self, output_path: Path) -> None:
        """Save analysis to JSON file.

        Args:
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 70)
        print("PERFORMANCE ANALYSIS")
        print("=" * 70)
        print(f"Throughput: {self.throughput_mbps:.2f} MB/s")
        print(f"Target: 5.0 MB/s")
        print(f"Status: {'✓ MEETS TARGET' if self.meets_target else '✗ BELOW TARGET'}")
        print(f"Documents: {self.documents_processed}")
        print(f"Data Processed: {self.total_mb_processed:.2f} MB")
        print(f"Avg Time/Doc: {self.avg_processing_time_ms:.2f}ms")
        print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Production Readiness Checklist
# ---------------------------------------------------------------------------


@dataclass
class ProductionReadinessChecklist:
    """Production readiness validation checklist.

    Tracks all quality gates from the production plan.

    Attributes:
        format_coverage_complete: All formats tested successfully
        determinism_tests_pass: Determinism tests 100% passing
        performance_meets_target: Throughput ≥5 MB/s
        memory_under_limit: Memory usage <2GB for large files
        integration_tests_pass: All integration tests passing
        quarantine_workflow_tested: Quarantine handles all failure modes
        privacy_audit_clean: No content leakage detected
        streaming_handles_large_files: >1GB files processed without OOM
        offline_operation_verified: No network calls during normalization
        metrics_exported_correctly: Telemetry data complete and accurate
        checklist_items: Detailed status of each item
        timestamp: Checklist generation timestamp
    """

    format_coverage_complete: bool = False
    determinism_tests_pass: bool = False
    performance_meets_target: bool = False
    memory_under_limit: bool = False
    integration_tests_pass: bool = False
    quarantine_workflow_tested: bool = False
    privacy_audit_clean: bool = False
    streaming_handles_large_files: bool = False
    offline_operation_verified: bool = False
    metrics_exported_correctly: bool = False

    checklist_items: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def set_item_status(
        self,
        item_name: str,
        passed: bool,
        details: Optional[str] = None
    ) -> None:
        """Set status for a checklist item.

        Args:
            item_name: Name of checklist item
            passed: Whether item passed
            details: Optional details/notes
        """
        self.checklist_items[item_name] = {
            "passed": passed,
            "details": details,
            "checked_at": datetime.now(timezone.utc).isoformat()
        }

        # Update corresponding boolean field
        field_map = {
            "format_coverage": "format_coverage_complete",
            "determinism": "determinism_tests_pass",
            "performance": "performance_meets_target",
            "memory": "memory_under_limit",
            "integration": "integration_tests_pass",
            "quarantine": "quarantine_workflow_tested",
            "privacy": "privacy_audit_clean",
            "streaming": "streaming_handles_large_files",
            "offline": "offline_operation_verified",
            "metrics": "metrics_exported_correctly"
        }

        for key, attr in field_map.items():
            if key in item_name.lower():
                setattr(self, attr, passed)

    @property
    def all_items_passing(self) -> bool:
        """Check if all checklist items are passing.

        Returns:
            True if all items pass, False otherwise
        """
        return all([
            self.format_coverage_complete,
            self.determinism_tests_pass,
            self.performance_meets_target,
            self.memory_under_limit,
            self.integration_tests_pass,
            self.quarantine_workflow_tested,
            self.privacy_audit_clean,
            self.streaming_handles_large_files,
            self.offline_operation_verified,
            self.metrics_exported_correctly
        ])

    @property
    def passing_percentage(self) -> float:
        """Calculate percentage of passing items.

        Returns:
            Percentage (0-100) of items passing
        """
        items = [
            self.format_coverage_complete,
            self.determinism_tests_pass,
            self.performance_meets_target,
            self.memory_under_limit,
            self.integration_tests_pass,
            self.quarantine_workflow_tested,
            self.privacy_audit_clean,
            self.streaming_handles_large_files,
            self.offline_operation_verified,
            self.metrics_exported_correctly
        ]
        return (sum(items) / len(items)) * 100

    def to_dict(self) -> dict:
        """Export checklist as dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "all_passing": self.all_items_passing,
                "passing_percentage": round(self.passing_percentage, 2)
            },
            "core_checks": {
                "format_coverage_complete": self.format_coverage_complete,
                "determinism_tests_pass": self.determinism_tests_pass,
                "performance_meets_target": self.performance_meets_target,
                "memory_under_limit": self.memory_under_limit,
                "integration_tests_pass": self.integration_tests_pass,
                "quarantine_workflow_tested": self.quarantine_workflow_tested,
                "privacy_audit_clean": self.privacy_audit_clean,
                "streaming_handles_large_files": self.streaming_handles_large_files,
                "offline_operation_verified": self.offline_operation_verified,
                "metrics_exported_correctly": self.metrics_exported_correctly
            },
            "detailed_items": self.checklist_items
        }

    def save_json(self, output_path: Path) -> None:
        """Save checklist to JSON file.

        Args:
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 70)
        print("PRODUCTION READINESS CHECKLIST")
        print("=" * 70)
        print(f"Overall Status: {self.passing_percentage:.1f}% Complete")
        print(f"Production Ready: {'✓ YES' if self.all_items_passing else '✗ NO'}")
        print("=" * 70)
        print("\nCore Checks:")
        print(f"  {'✓' if self.format_coverage_complete else '✗'} All 16+ formats parse successfully")
        print(f"  {'✓' if self.determinism_tests_pass else '✗'} Determinism tests pass 100%")
        print(f"  {'✓' if self.performance_meets_target else '✗'} Performance ≥5 MB/s")
        print(f"  {'✓' if self.memory_under_limit else '✗'} Memory usage <2GB for large files")
        print(f"  {'✓' if self.integration_tests_pass else '✗'} Integration tests pass")
        print(f"  {'✓' if self.quarantine_workflow_tested else '✗'} Quarantine workflow handles failures")
        print(f"  {'✓' if self.privacy_audit_clean else '✗'} Privacy audit clean (no leaks)")
        print(f"  {'✓' if self.streaming_handles_large_files else '✗'} Streaming handles >1GB files")
        print(f"  {'✓' if self.offline_operation_verified else '✗'} Offline operation verified")
        print(f"  {'✓' if self.metrics_exported_correctly else '✗'} Metrics exported correctly")
        print("=" * 70 + "\n")


def create_production_readiness_checklist() -> ProductionReadinessChecklist:
    """Create a new production readiness checklist.

    Returns:
        Empty ProductionReadinessChecklist instance
    """
    return ProductionReadinessChecklist()
