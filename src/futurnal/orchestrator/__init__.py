"""Ingestion orchestrator package."""

from .queue import JobStatus
from .scheduler import IngestionOrchestrator
from .health import collect_health_report
from .quarantine import QuarantineStore, QuarantineReason, QuarantinedJob, classify_failure
from .resource_profile import (
    ResourceIntensity,
    IOPattern,
    ResourceProfile,
    JobResourceMetrics,
    ConnectorResourceStats,
)
from .resource_monitor import ResourceMonitor
from .resource_registry import ResourceProfileRegistry
from .status import collect_status_report
from .source_control import PausedSourcesRegistry
from .exceptions import (
    InvalidStateTransitionError,
    StateTransitionRaceError,
    JobNotFoundError,
)
from .state_machine import (
    StateTransition,
    StateMachineValidator,
    StateMachineInvariants,
    VALID_TRANSITIONS,
)
from .deadlock import DeadlockDetector
from .integrity import validate_database_integrity
from .crash_recovery import (
    CrashRecoveryReport,
    RecoveryStateTracker,
    CrashRecoveryManager,
)

__all__ = [
    "JobStatus",
    "IngestionOrchestrator",
    "collect_health_report",
    "collect_status_report",
    "QuarantineStore",
    "QuarantineReason",
    "QuarantinedJob",
    "classify_failure",
    "ResourceIntensity",
    "IOPattern",
    "ResourceProfile",
    "JobResourceMetrics",
    "ConnectorResourceStats",
    "ResourceMonitor",
    "ResourceProfileRegistry",
    "PausedSourcesRegistry",
    "InvalidStateTransitionError",
    "StateTransitionRaceError",
    "JobNotFoundError",
    "StateTransition",
    "StateMachineValidator",
    "StateMachineInvariants",
    "VALID_TRANSITIONS",
    "DeadlockDetector",
    "validate_database_integrity",
    "CrashRecoveryReport",
    "RecoveryStateTracker",
    "CrashRecoveryManager",
]


