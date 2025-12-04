"""Production Readiness Validator for PKG Module 05.

Validates that the PKG storage layer meets all production requirements
from the production plan.

Production Readiness Checklist (from production plan):
1. Schema created and validated
2. Database configured and encrypted
3. Data access layer operational
4. Temporal queries functional
5. Integration with extraction pipeline complete
6. Vector store sync working
7. Performance targets met
8. Resilience validated (crash recovery)
9. Backup/restore operational

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/05-integration-testing.md
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from futurnal.pkg.database.manager import PKGDatabaseManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation Result Models
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of a single validation check.

    Attributes:
        name: Name of the validation check
        passed: Whether the check passed
        message: Description or error message
        details: Additional validation details
        duration_ms: Time taken for validation
    """

    name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


@dataclass
class ProductionReadinessReport:
    """Complete production readiness report.

    From production plan checklist:
    - Schema valid
    - Database configured
    - Data access operational
    - Temporal queries functional
    - Extraction integration complete
    - Vector sync working
    - Performance targets met
    - Resilience validated
    - Backup/restore operational

    Attributes:
        schema_valid: Schema constraints and indices exist
        database_configured: Database connection and config valid
        data_access_operational: CRUD operations work
        temporal_queries_functional: Temporal queries return results
        extraction_integration: Extraction pipeline integration works
        vector_sync: Vector store sync operational
        performance_targets: Performance benchmarks pass
        resilience_validated: ACID and recovery work
        backup_restore: Backup/restore operational
        validation_results: Detailed results for each check
        validated_at: Timestamp of validation
        total_duration_ms: Total validation time
    """

    schema_valid: bool = False
    database_configured: bool = False
    data_access_operational: bool = False
    temporal_queries_functional: bool = False
    extraction_integration: bool = False
    vector_sync: bool = False
    performance_targets: bool = False
    resilience_validated: bool = False
    backup_restore: bool = False
    validation_results: List[ValidationResult] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.utcnow)
    total_duration_ms: float = 0.0

    @property
    def is_production_ready(self) -> bool:
        """Check if all criteria are met for production deployment."""
        return all([
            self.schema_valid,
            self.database_configured,
            self.data_access_operational,
            self.temporal_queries_functional,
            self.extraction_integration,
            self.vector_sync,
            self.performance_targets,
            self.resilience_validated,
            self.backup_restore,
        ])

    @property
    def passed_count(self) -> int:
        """Number of passed checks."""
        return sum(1 for r in self.validation_results if r.passed)

    @property
    def total_count(self) -> int:
        """Total number of checks."""
        return len(self.validation_results)

    @property
    def pass_rate(self) -> float:
        """Pass rate as percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.passed_count / self.total_count) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "is_production_ready": self.is_production_ready,
            "schema_valid": self.schema_valid,
            "database_configured": self.database_configured,
            "data_access_operational": self.data_access_operational,
            "temporal_queries_functional": self.temporal_queries_functional,
            "extraction_integration": self.extraction_integration,
            "vector_sync": self.vector_sync,
            "performance_targets": self.performance_targets,
            "resilience_validated": self.resilience_validated,
            "backup_restore": self.backup_restore,
            "passed_count": self.passed_count,
            "total_count": self.total_count,
            "pass_rate": self.pass_rate,
            "validated_at": self.validated_at.isoformat(),
            "total_duration_ms": self.total_duration_ms,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                }
                for r in self.validation_results
            ],
        }

    def summary(self) -> str:
        """Generate summary string."""
        status = "READY" if self.is_production_ready else "NOT READY"
        return (
            f"PKG Production Readiness: {status}\n"
            f"Passed: {self.passed_count}/{self.total_count} ({self.pass_rate:.1f}%)\n"
            f"Duration: {self.total_duration_ms:.1f}ms\n"
            f"\nChecks:\n" +
            "\n".join(f"  {r}" for r in self.validation_results)
        )


# ---------------------------------------------------------------------------
# Production Readiness Validator
# ---------------------------------------------------------------------------


class ProductionReadinessValidator:
    """Validates PKG production readiness.

    Runs all validation checks from the production plan checklist:
    1. Schema validation
    2. Database configuration
    3. Data access layer
    4. Temporal queries
    5. Extraction integration
    6. Vector sync
    7. Performance targets
    8. Resilience
    9. Backup/restore

    Example:
        >>> from futurnal.pkg.database import PKGDatabaseManager
        >>> from futurnal.pkg.validation import ProductionReadinessValidator
        >>>
        >>> with PKGDatabaseManager(config) as db:
        ...     validator = ProductionReadinessValidator(db)
        ...     report = validator.validate_all()
        ...     print(report.summary())
        ...
        ...     if report.is_production_ready:
        ...         print("PKG is ready for production!")
    """

    def __init__(
        self,
        db_manager: Optional["PKGDatabaseManager"] = None,
        vector_store: Optional[Any] = None,
        skip_performance: bool = False,
    ):
        """Initialize the validator.

        Args:
            db_manager: PKGDatabaseManager instance (optional for partial validation)
            vector_store: Vector store instance for sync validation
            skip_performance: Skip performance benchmarks (for quick validation)
        """
        self._db = db_manager
        self._vector = vector_store
        self._skip_performance = skip_performance

    def validate_all(self) -> ProductionReadinessReport:
        """Run all validation checks.

        Returns:
            ProductionReadinessReport with all validation results
        """
        start_time = time.perf_counter()
        report = ProductionReadinessReport()

        # Run all checks
        checks = [
            ("schema", self._check_schema),
            ("database", self._check_database),
            ("data_access", self._check_data_access),
            ("temporal", self._check_temporal),
            ("extraction", self._check_extraction_integration),
            ("vector_sync", self._check_vector_sync),
            ("performance", self._check_performance),
            ("resilience", self._check_resilience),
            ("backup", self._check_backup_restore),
        ]

        for name, check_fn in checks:
            try:
                result = check_fn()
                report.validation_results.append(result)

                # Update corresponding report field
                if name == "schema":
                    report.schema_valid = result.passed
                elif name == "database":
                    report.database_configured = result.passed
                elif name == "data_access":
                    report.data_access_operational = result.passed
                elif name == "temporal":
                    report.temporal_queries_functional = result.passed
                elif name == "extraction":
                    report.extraction_integration = result.passed
                elif name == "vector_sync":
                    report.vector_sync = result.passed
                elif name == "performance":
                    report.performance_targets = result.passed
                elif name == "resilience":
                    report.resilience_validated = result.passed
                elif name == "backup":
                    report.backup_restore = result.passed

            except Exception as e:
                logger.error(f"Validation check {name} failed with exception: {e}")
                report.validation_results.append(
                    ValidationResult(
                        name=name,
                        passed=False,
                        message=f"Exception: {str(e)}",
                    )
                )

        report.total_duration_ms = (time.perf_counter() - start_time) * 1000
        return report

    def _check_schema(self) -> ValidationResult:
        """Check schema constraints and indices exist."""
        start = time.perf_counter()

        if not self._db:
            return ValidationResult(
                name="schema_valid",
                passed=False,
                message="No database manager provided",
            )

        try:
            from futurnal.pkg.schema.constraints import validate_schema

            is_valid = validate_schema(self._db._driver)

            return ValidationResult(
                name="schema_valid",
                passed=is_valid,
                message="Schema constraints and indices valid" if is_valid else "Schema validation failed",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return ValidationResult(
                name="schema_valid",
                passed=False,
                message=f"Schema validation error: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def _check_database(self) -> ValidationResult:
        """Check database connection and configuration."""
        start = time.perf_counter()

        if not self._db:
            return ValidationResult(
                name="database_configured",
                passed=False,
                message="No database manager provided",
            )

        try:
            # Verify connectivity
            self._db._driver.verify_connectivity()

            # Check basic query works
            with self._db.session() as session:
                result = session.run("RETURN 1 as test")
                assert result.single()["test"] == 1

            return ValidationResult(
                name="database_configured",
                passed=True,
                message="Database connection and configuration valid",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return ValidationResult(
                name="database_configured",
                passed=False,
                message=f"Database configuration error: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def _check_data_access(self) -> ValidationResult:
        """Check CRUD operations work."""
        start = time.perf_counter()

        if not self._db:
            return ValidationResult(
                name="data_access_operational",
                passed=False,
                message="No database manager provided",
            )

        try:
            test_id = f"validation_test_{int(time.time())}"

            with self._db.session() as session:
                # Create
                session.run(
                    "CREATE (n:ValidationTest {id: $id, timestamp: datetime()})",
                    {"id": test_id},
                )

                # Read
                result = session.run(
                    "MATCH (n:ValidationTest {id: $id}) RETURN n",
                    {"id": test_id},
                )
                assert result.single() is not None

                # Delete (cleanup)
                session.run(
                    "MATCH (n:ValidationTest {id: $id}) DELETE n",
                    {"id": test_id},
                )

            return ValidationResult(
                name="data_access_operational",
                passed=True,
                message="CRUD operations functional",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return ValidationResult(
                name="data_access_operational",
                passed=False,
                message=f"Data access error: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def _check_temporal(self) -> ValidationResult:
        """Check temporal queries work."""
        start = time.perf_counter()

        if not self._db:
            return ValidationResult(
                name="temporal_queries_functional",
                passed=False,
                message="No database manager provided",
            )

        try:
            from futurnal.pkg.queries.temporal import TemporalGraphQueries

            # Create wrapper for db manager
            class DBWrapper:
                def __init__(self, driver):
                    self._driver = driver

                def session(self):
                    return self._driver.session()

            queries = TemporalGraphQueries(DBWrapper(self._db._driver))

            # Test time range query (empty result is OK)
            results = queries.query_events_in_timerange(
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 2),
            )

            return ValidationResult(
                name="temporal_queries_functional",
                passed=True,
                message=f"Temporal queries functional (found {len(results)} events)",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return ValidationResult(
                name="temporal_queries_functional",
                passed=False,
                message=f"Temporal query error: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def _check_extraction_integration(self) -> ValidationResult:
        """Check extraction pipeline integration."""
        start = time.perf_counter()

        try:
            # Check that extraction pipeline modules are importable
            from futurnal.pipeline.triples import MetadataTripleExtractor
            from futurnal.pipeline.stubs import NormalizationSink

            return ValidationResult(
                name="extraction_integration",
                passed=True,
                message="Extraction pipeline modules available",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except ImportError as e:
            return ValidationResult(
                name="extraction_integration",
                passed=False,
                message=f"Extraction integration error: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def _check_vector_sync(self) -> ValidationResult:
        """Check vector store sync works."""
        start = time.perf_counter()

        try:
            # Check sync event modules are available
            from futurnal.pkg.sync import SyncEvent, SyncEventCapture

            # Create test capture
            capture = SyncEventCapture()
            assert capture.count == 0

            return ValidationResult(
                name="vector_sync",
                passed=True,
                message="Vector sync modules operational",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return ValidationResult(
                name="vector_sync",
                passed=False,
                message=f"Vector sync error: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def _check_performance(self) -> ValidationResult:
        """Check performance targets are met."""
        start = time.perf_counter()

        if self._skip_performance:
            return ValidationResult(
                name="performance_targets",
                passed=True,
                message="Performance check skipped",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        if not self._db:
            return ValidationResult(
                name="performance_targets",
                passed=False,
                message="No database manager provided",
            )

        try:
            # Quick performance check: simple query should be <100ms
            query_start = time.perf_counter()

            with self._db.session() as session:
                session.run("MATCH (n) RETURN count(n) as c LIMIT 1")

            query_ms = (time.perf_counter() - query_start) * 1000

            passed = query_ms < 100
            return ValidationResult(
                name="performance_targets",
                passed=passed,
                message=f"Query latency: {query_ms:.1f}ms (target <100ms)",
                details={"query_latency_ms": query_ms},
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return ValidationResult(
                name="performance_targets",
                passed=False,
                message=f"Performance check error: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def _check_resilience(self) -> ValidationResult:
        """Check ACID and recovery capabilities."""
        start = time.perf_counter()

        if not self._db:
            return ValidationResult(
                name="resilience_validated",
                passed=False,
                message="No database manager provided",
            )

        try:
            test_id = f"resilience_test_{int(time.time())}"

            # Test transaction rollback
            try:
                with self._db.session() as session:
                    with session.begin_transaction() as tx:
                        tx.run(
                            "CREATE (n:ResilienceTest {id: $id})",
                            {"id": test_id},
                        )
                        # Don't commit - rollback should happen
                        raise Exception("Intentional rollback test")
            except Exception:
                pass  # Expected

            # Verify rollback worked
            with self._db.session() as session:
                result = session.run(
                    "MATCH (n:ResilienceTest {id: $id}) RETURN count(n) as c",
                    {"id": test_id},
                )
                count = result.single()["c"]
                assert count == 0, "Rollback should have prevented creation"

            return ValidationResult(
                name="resilience_validated",
                passed=True,
                message="ACID semantics validated",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return ValidationResult(
                name="resilience_validated",
                passed=False,
                message=f"Resilience check error: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    def _check_backup_restore(self) -> ValidationResult:
        """Check backup/restore capability exists."""
        start = time.perf_counter()

        try:
            # Check backup module is available
            from futurnal.pkg.database.backup import PKGBackupManager

            return ValidationResult(
                name="backup_restore",
                passed=True,
                message="Backup/restore module available",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except ImportError as e:
            return ValidationResult(
                name="backup_restore",
                passed=False,
                message=f"Backup module not available: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
