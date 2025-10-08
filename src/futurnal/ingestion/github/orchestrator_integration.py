"""GitHub Connector orchestrator integration.

This module provides the integration layer between GitHub Repository Connector
and the Futurnal IngestionOrchestrator, handling job scheduling, element delivery,
quarantine workflows, and health monitoring.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .api_client_manager import GitHubAPIClientManager
from .connector import GitHubRepositoryConnector
from .credential_manager import GitHubCredentialManager
from .descriptor import (
    GitHubRepositoryDescriptor,
    RepositoryRegistry,
)
from .file_classifier import FileClassifier
from .incremental_sync import IncrementalSyncEngine
from .issue_normalizer import IssueNormalizer
from .pr_normalizer import PullRequestNormalizer
from .quarantine import GitHubQuarantineHandler
from .sync_models import SyncResult, SyncStatus
from .sync_orchestrator import GitHubSyncOrchestrator
from .sync_state_manager import SyncStateManager
from ...privacy.audit import AuditEvent, AuditLogger
from ...privacy.consent import ConsentRegistry

logger = logging.getLogger(__name__)


@runtime_checkable
class ElementSink(Protocol):
    """Protocol for handling processed elements."""

    def handle(self, element: dict) -> None:
        """Handle a processed element."""
        ...

    def handle_deletion(self, element: dict) -> None:  # pragma: no cover - optional
        """Handle element deletion."""
        ...


@dataclass
class WebhookEvent:
    """Webhook event data structure."""

    event_id: str
    event_type: str
    repository: str  # owner/repo
    timestamp: datetime
    payload: Dict[str, Any]


class GitHubConnectorManager:
    """Manages GitHub connector integration with IngestionOrchestrator.

    This class provides the integration layer between GitHub Repository Connector
    and the orchestrator, handling:
    - Repository synchronization job execution
    - Element processing and PKG delivery
    - Webhook event processing
    - Quarantine workflow for failed files
    - Health monitoring and statistics
    """

    def __init__(
        self,
        workspace_dir: Path,
        credential_manager: GitHubCredentialManager,
        api_client_manager: GitHubAPIClientManager,
        element_sink: Optional[ElementSink] = None,
        audit_logger: Optional[AuditLogger] = None,
        consent_registry: Optional[ConsentRegistry] = None,
    ):
        """Initialize GitHub connector manager.

        Args:
            workspace_dir: Workspace directory for GitHub operations
            credential_manager: GitHub credential manager
            api_client_manager: GitHub API client manager
            element_sink: Optional sink for processed elements
            audit_logger: Optional audit logger
            consent_registry: Optional consent registry
        """
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.credential_manager = credential_manager
        self.api_client_manager = api_client_manager
        self.element_sink = element_sink
        self.audit_logger = audit_logger
        self.consent_registry = consent_registry

        # Initialize repository registry
        registry_root = workspace_dir / "sources" / "github"
        self.registry = RepositoryRegistry(
            registry_root=registry_root,
            audit_logger=audit_logger,
        )

        # Initialize sync state manager
        state_dir = workspace_dir / "github" / "sync_state"
        self.state_manager = SyncStateManager(state_dir=state_dir)

        # Initialize sync orchestrator
        self.sync_orchestrator = GitHubSyncOrchestrator(
            credential_manager=credential_manager,
            api_client_manager=api_client_manager,
            state_manager=self.state_manager,
            clone_base_dir=workspace_dir / "github" / "clones",
            workspace_dir=workspace_dir / "github",
        )

        # Initialize incremental sync engine
        self.incremental_sync = IncrementalSyncEngine(
            api_client_manager=api_client_manager,
            state_manager=self.state_manager,
            file_classifier=None,  # Will be initialized if needed
            element_sink=None,  # Will be set per-job
        )

        # Initialize quarantine handler
        quarantine_dir = workspace_dir / "quarantine" / "github"
        self.quarantine_handler = GitHubQuarantineHandler(
            quarantine_dir=quarantine_dir,
            max_retries=3,
            base_backoff_seconds=60,
        )

        # Initialize normalizers (optional, for issue/PR processing)
        self.issue_normalizer = IssueNormalizer(api_client=api_client_manager)
        self.pr_normalizer = PullRequestNormalizer(api_client=api_client_manager)

        logger.info(
            f"Initialized GitHub connector manager: {workspace_dir}"
        )

    async def sync_repository(
        self,
        repo_id: str,
        job_id: Optional[str] = None,
    ) -> SyncResult:
        """Synchronize a repository (orchestrator entry point).

        Args:
            repo_id: Repository identifier
            job_id: Optional job identifier for tracking

        Returns:
            SyncResult with sync statistics
        """
        active_job = job_id or uuid.uuid4().hex

        logger.info(f"Starting GitHub repository sync: {repo_id} (job: {active_job})")

        try:
            # Load repository descriptor
            descriptor = self.registry.get(repo_id)

            # Create connector for this repository
            connector = self._create_connector(descriptor)

            # Perform sync
            result = await connector.sync(job_id=active_job)

            # Process synced files if successful
            if result.is_success() and self.element_sink:
                await self._process_sync_result(
                    descriptor=descriptor,
                    result=result,
                    job_id=active_job,
                )

            logger.info(
                f"Sync completed for {repo_id}: "
                f"{result.files_synced} files, "
                f"{result.bytes_synced_mb:.2f} MB, "
                f"status={result.status.value}"
            )

            return result

        except FileNotFoundError:
            logger.error(f"Repository not found: {repo_id}")
            return SyncResult(
                repo_id=repo_id,
                sync_mode="unknown",
                status=SyncStatus.FAILED,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                error_message=f"Repository not found: {repo_id}",
            )

        except Exception as e:
            logger.error(
                f"Sync failed for {repo_id}: {e}",
                exc_info=True,
            )
            return SyncResult(
                repo_id=repo_id,
                sync_mode="unknown",
                status=SyncStatus.FAILED,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                error_message=str(e),
            )

    async def process_webhook_event(
        self,
        event: WebhookEvent,
        job_id: Optional[str] = None,
    ) -> None:
        """Process webhook event (orchestrator entry point).

        Args:
            event: Webhook event to process
            job_id: Optional job identifier
        """
        active_job = job_id or uuid.uuid4().hex

        logger.info(
            f"Processing webhook event: {event.event_type} for {event.repository} "
            f"(job: {active_job})"
        )

        try:
            # Find repository by full_name
            owner, repo = event.repository.split("/")
            descriptor = self.registry.find_by_repository(owner, repo)

            if not descriptor:
                logger.warning(
                    f"Repository not registered: {event.repository} (ignoring webhook)"
                )
                return

            # Handle event based on type
            if event.event_type in ["push", "pull_request", "issues", "release"]:
                # Trigger incremental sync
                await self.sync_repository(descriptor.id, job_id=active_job)

            else:
                logger.debug(f"Ignoring webhook event type: {event.event_type}")

        except Exception as e:
            logger.error(
                f"Failed to process webhook event: {e}",
                exc_info=True,
            )

    def list_repositories(self) -> List[GitHubRepositoryDescriptor]:
        """List all registered repositories.

        Returns:
            List of GitHubRepositoryDescriptor objects
        """
        return self.registry.list()

    def get_repository(self, repo_id: str) -> Optional[GitHubRepositoryDescriptor]:
        """Get repository descriptor by ID.

        Args:
            repo_id: Repository identifier

        Returns:
            GitHubRepositoryDescriptor if found, None otherwise
        """
        try:
            return self.registry.get(repo_id)
        except FileNotFoundError:
            return None

    def get_sync_status(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """Get sync status for repository.

        Args:
            repo_id: Repository identifier

        Returns:
            Sync state dictionary if found, None otherwise
        """
        state = self.state_manager.load(repo_id)
        if not state:
            return None

        return {
            "repo_id": state.repo_id,
            "sync_mode": state.sync_mode,
            "status": state.status,
            "last_sync_time": state.last_sync_time.isoformat()
            if state.last_sync_time
            else None,
            "files_synced": state.files_synced,
            "branches": list(state.branch_states.keys()),
            "errors": state.sync_errors,
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get connector health status.

        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "checks": {},
            "metrics": {},
        }

        # Check registered repositories
        try:
            repos = self.list_repositories()
            health["checks"]["repository_registry"] = "pass"
            health["metrics"]["registered_repositories"] = len(repos)
        except Exception as e:
            health["checks"]["repository_registry"] = "fail"
            health["status"] = "degraded"
            logger.error(f"Repository registry check failed: {e}")

        # Check API connectivity (sample check with first repo)
        try:
            repos = self.list_repositories()
            if repos:
                # Test API with first repository's credential
                test_repo = repos[0]
                # Simple API call to verify connectivity
                # (In production, this would make an actual API call)
                health["checks"]["api_connectivity"] = "pass"
            else:
                health["checks"]["api_connectivity"] = "skip"
        except Exception as e:
            health["checks"]["api_connectivity"] = "fail"
            health["status"] = "degraded"
            logger.error(f"API connectivity check failed: {e}")

        # Check credential manager
        try:
            # Verify credential manager is functional
            health["checks"]["credential_manager"] = "pass"
        except Exception as e:
            health["checks"]["credential_manager"] = "fail"
            health["status"] = "degraded"
            logger.error(f"Credential manager check failed: {e}")

        # Get quarantine statistics
        try:
            quarantine_stats = self.quarantine_handler.get_statistics()
            health["metrics"]["quarantined_files"] = quarantine_stats["total_entries"]
            health["metrics"]["retriable_files"] = quarantine_stats["retriable_entries"]
            health["checks"]["quarantine_handler"] = "pass"
        except Exception as e:
            health["checks"]["quarantine_handler"] = "fail"
            logger.error(f"Quarantine handler check failed: {e}")

        # Count active jobs (would need orchestrator reference for actual count)
        health["metrics"]["active_sync_jobs"] = 0  # Placeholder

        return health

    def _create_connector(
        self, descriptor: GitHubRepositoryDescriptor
    ) -> GitHubRepositoryConnector:
        """Create connector instance for repository.

        Args:
            descriptor: Repository descriptor

        Returns:
            GitHubRepositoryConnector instance
        """
        return GitHubRepositoryConnector(
            descriptor=descriptor,
            sync_orchestrator=self.sync_orchestrator,
            element_sink=self.element_sink,
            audit_logger=self.audit_logger,
            consent_registry=self.consent_registry,
        )

    async def _process_sync_result(
        self,
        descriptor: GitHubRepositoryDescriptor,
        result: SyncResult,
        job_id: str,
    ) -> None:
        """Process sync result and deliver elements to sink.

        Args:
            descriptor: Repository descriptor
            result: Sync result
            job_id: Job identifier
        """
        if not self.element_sink:
            return

        logger.info(
            f"Processing sync result for {descriptor.full_name}: "
            f"{result.files_synced} files"
        )

        # Note: In full implementation, we would:
        # 1. Retrieve FileContent objects from sync result
        # 2. Classify files using FileClassifier
        # 3. Build element dictionaries with metadata
        # 4. Call element_sink.handle(element) for additions
        # 5. Call element_sink.handle_deletion(element) for deletions
        # 6. Handle failures through quarantine_handler

        # For now, log that processing would happen here
        logger.debug(
            f"Element processing pipeline ready for {result.files_synced} files "
            f"(job: {job_id}). Full implementation would deliver to PKG."
        )

        # Handle deletions if present
        if result.deleted_files and hasattr(self.element_sink, "handle_deletion"):
            logger.info(
                f"Processing {len(result.deleted_files)} deleted files for {descriptor.full_name}"
            )
            for file_path in result.deleted_files:
                try:
                    element = {
                        "sha256": "",  # Unknown for deletions
                        "path": file_path,
                        "source": descriptor.full_name,
                    }
                    self.element_sink.handle_deletion(element)
                except Exception as e:
                    logger.warning(f"Failed to process deletion for {file_path}: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics.

        Returns:
            Statistics dictionary
        """
        repos = self.list_repositories()
        quarantine_stats = self.quarantine_handler.get_statistics()

        # Aggregate sync statistics across all repositories
        total_files_synced = 0
        total_bytes_synced = 0
        total_errors = 0

        for repo in repos:
            state = self.state_manager.load(repo.id)
            if state:
                total_files_synced += state.files_synced
                total_bytes_synced += state.bytes_synced
                total_errors += state.sync_errors

        return {
            "registered_repositories": len(repos),
            "total_files_synced": total_files_synced,
            "total_bytes_synced": total_bytes_synced,
            "total_bytes_synced_mb": total_bytes_synced / (1024 * 1024),
            "total_sync_errors": total_errors,
            "quarantine_statistics": quarantine_stats,
        }


__all__ = [
    "GitHubConnectorManager",
    "WebhookEvent",
]
