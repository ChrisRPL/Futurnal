"""PKG Validation Module - Production Readiness Validation.

Provides validators for verifying PKG production readiness.

Module Structure:
- readiness.py: ProductionReadinessValidator and reports

Implementation follows production plan:
docs/phase-1/pkg-graph-storage-production-plan/05-integration-testing.md

Usage:
    >>> from futurnal.pkg.validation import ProductionReadinessValidator
    >>>
    >>> # Validate production readiness
    >>> validator = ProductionReadinessValidator(db_manager, vector_store)
    >>> report = validator.validate_all()
    >>> print(f"Production ready: {report.is_production_ready}")
"""

from futurnal.pkg.validation.readiness import (
    ProductionReadinessValidator,
    ProductionReadinessReport,
    ValidationResult,
)

__all__ = [
    "ProductionReadinessValidator",
    "ProductionReadinessReport",
    "ValidationResult",
]
