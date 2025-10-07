Summary: Integrate GitHub connector with IngestionOrchestrator for scheduled syncs and job management.

# 10 · Orchestrator Integration

## Purpose
Integrate the GitHub Repository Connector with Futurnal's IngestionOrchestrator for scheduled synchronization, job queue management, and element delivery to PKG/vector stores. Enable both polling-based and webhook-triggered sync operations.

## Scope
- Connector registration with IngestionOrchestrator
- Scheduled sync jobs via APScheduler
- Webhook event queue integration
- ElementSink integration for PKG population
- StateStore integration for sync checkpoints
- Quarantine workflow for failed processing
- Job priority and resource management
- Health monitoring and metrics

## Requirements Alignment
- **Scheduled execution**: Periodic syncs every 5-15 minutes (configurable)
- **Event-driven**: Webhook events trigger immediate syncs
- **Resilience**: Failed jobs quarantined with retry policies
- **Resource management**: Respect system resources and API limits
- **Observability**: Job status, metrics, and health monitoring

## Data Model

### GitHubConnectorJob
```python
class GitHubConnectorJob(BaseModel):
    """Job definition for GitHub connector."""

    job_id: str
    job_type: str  # "full_sync", "incremental_sync", "webhook_event"
    repo_id: str
    repo_descriptor: GitHubRepositoryDescriptor

    # Scheduling
    schedule: Optional[str] = None  # Cron expression
    priority: int = 5  # 1 (highest) to 10 (lowest)

    # Execution
    status: str = "pending"  # "pending", "running", "completed", "failed"
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    result: Optional[SyncResult] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
```

## Component Design

### GitHubRepositoryConnector (Main)
```python
class GitHubRepositoryConnector:
    """Main connector integrating all GitHub components."""

    def __init__(
        self,
        *,
        workspace_dir: Path,
        credential_manager: GitHubCredentialManager,
        api_client_manager: GitHubAPIClientManager,
        file_classifier: FileClassifier,
        issue_normalizer: IssueNormalizer,
        pr_normalizer: PullRequestNormalizer,
        sync_engine: IncrementalSyncEngine,
        element_sink: ElementSink,
        state_store: StateStore,
        audit_logger: AuditLogger,
        consent_registry: ConsentRegistry,
    ):
        self.workspace_dir = workspace_dir
        self.credential_manager = credential_manager
        self.api_client_manager = api_client_manager
        self.file_classifier = file_classifier
        self.issue_normalizer = issue_normalizer
        self.pr_normalizer = pr_normalizer
        self.sync_engine = sync_engine
        self.element_sink = element_sink
        self.state_store = state_store
        self.audit_logger = audit_logger
        self.consent_registry = consent_registry

        # Repository registry
        self.registry = RepositoryRegistry(workspace_dir / "sources" / "github")

    async def sync_repository(
        self,
        repo_id: str,
        job_id: Optional[str] = None,
    ) -> SyncResult:
        """Sync a repository (orchestrator entry point)."""

        # Load repository descriptor
        descriptor = self.registry.get(repo_id)
        if not descriptor:
            raise ValueError(f"Repository not found: {repo_id}")

        # Check consent
        with require_consent(
            self.consent_registry,
            repo_id,
            GitHubConsentScope.GITHUB_REPO_ACCESS,
        ):
            # Perform sync
            result = await self.sync_engine.sync_repository(descriptor)

            # Audit log
            self.audit_logger.log_repository_sync(
                repo_id=repo_id,
                repo_full_name=descriptor.full_name,
                branch=descriptor.branches[0],  # Primary branch
                result=result,
                status="success",
            )

            return result

    async def process_webhook_event(
        self,
        event: WebhookEvent,
        job_id: Optional[str] = None,
    ) -> None:
        """Process webhook event (orchestrator entry point)."""

        # Extract repo_id from event
        repo_full_name = event.repository
        descriptor = self.registry.get_by_full_name(repo_full_name)

        if not descriptor:
            logger.warning(f"Repository not registered: {repo_full_name}")
            return

        # Handle event based on type
        if event.event_type == WebhookEventType.PUSH:
            # Trigger incremental sync
            await self.sync_repository(descriptor.id, job_id)

        elif event.event_type == WebhookEventType.ISSUES:
            # Update issue metadata
            await self._sync_issue_metadata(descriptor, event.payload)

        elif event.event_type == WebhookEventType.PULL_REQUEST:
            # Update PR metadata
            await self._sync_pr_metadata(descriptor, event.payload)

    def list_repositories(self) -> List[GitHubRepositoryDescriptor]:
        """List all registered repositories."""
        return self.registry.list_all()

    def get_repository(self, repo_id: str) -> Optional[GitHubRepositoryDescriptor]:
        """Get repository descriptor."""
        return self.registry.get(repo_id)
```

### Orchestrator Registration
```python
def register_github_connector(
    orchestrator: IngestionOrchestrator,
    connector: GitHubRepositoryConnector,
):
    """Register GitHub connector with orchestrator."""

    # Register connector type
    orchestrator.register_connector(
        connector_type="github_repository",
        connector_instance=connector,
        sync_method=connector.sync_repository,
    )

    # Register webhook handler
    if hasattr(connector, 'webhook_server'):
        orchestrator.register_webhook_handler(
            connector_type="github_repository",
            handler=connector.process_webhook_event,
        )

    # Schedule sync jobs for all repositories
    for descriptor in connector.list_repositories():
        _schedule_repository_sync(orchestrator, descriptor)

def _schedule_repository_sync(
    orchestrator: IngestionOrchestrator,
    descriptor: GitHubRepositoryDescriptor,
):
    """Schedule periodic sync job for repository."""

    # Default: sync every 5 minutes if webhooks disabled
    schedule_interval = "*/5 * * * *"  # Every 5 minutes

    if hasattr(descriptor, 'webhook_enabled') and descriptor.webhook_enabled:
        # With webhooks, sync less frequently (hourly verification)
        schedule_interval = "0 * * * *"  # Every hour

    job = GitHubConnectorJob(
        job_id=f"github_sync_{descriptor.id}",
        job_type="incremental_sync",
        repo_id=descriptor.id,
        repo_descriptor=descriptor,
        schedule=schedule_interval,
        priority=5,
    )

    orchestrator.schedule_job(job)
```

## ElementSink Integration

### Element Delivery
```python
async def deliver_elements_to_sink(
    sync_result: SyncResult,
    file_contents: List[FileContent],
    element_sink: ElementSink,
):
    """Deliver processed elements to PKG/vector stores."""

    for file_content in file_contents:
        # Classify file
        classification = file_classifier.classify_file(
            file_content.path,
            file_content.content,
        )

        # Generate elements via Unstructured.io
        elements = partition(
            text=file_content.content,
            content_type=classification.category.value,
        )

        # Deliver to sink
        for element in elements:
            element_dict = {
                "element_id": element.id,
                "type": element.category,
                "text": element.text,
                "metadata": {
                    "repo_id": sync_result.repo_id,
                    "branch": sync_result.branch,
                    "file_path": file_content.path,
                    "file_category": classification.category.value,
                    "language": classification.language,
                    "commit_sha": file_content.commit_sha,
                },
            }

            element_sink.handle(element_dict)
```

## StateStore Integration

### Sync State Persistence
```python
class GitHubStateStore:
    """State store adapter for GitHub connector."""

    def __init__(self, base_state_store: StateStore):
        self.store = base_state_store
        self.prefix = "github:"

    def save_sync_state(
        self,
        repo_id: str,
        state: RepositorySyncState,
    ):
        """Save repository sync state."""
        key = f"{self.prefix}sync:{repo_id}"
        self.store.set(key, state.model_dump_json())

    def load_sync_state(
        self,
        repo_id: str,
    ) -> Optional[RepositorySyncState]:
        """Load repository sync state."""
        key = f"{self.prefix}sync:{repo_id}"
        data = self.store.get(key)

        if data:
            return RepositorySyncState.model_validate_json(data)
        return None

    def save_webhook_event(
        self,
        event: WebhookEvent,
    ):
        """Save webhook event for processing."""
        key = f"{self.prefix}webhook:{event.event_id}"
        self.store.set(key, event.model_dump_json())
```

## Quarantine Workflow

### Failed Processing Handler
```python
class GitHubQuarantineHandler:
    """Handles failed GitHub processing with retry policies."""

    def __init__(
        self,
        quarantine_dir: Path,
        max_retries: int = 3,
    ):
        self.quarantine_dir = quarantine_dir
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries

    async def quarantine_failed_file(
        self,
        repo_id: str,
        file_path: str,
        content: bytes,
        error: Exception,
        retry_count: int,
    ):
        """Quarantine file that failed processing."""

        # Create quarantine entry
        quarantine_id = f"{repo_id}_{sha256(file_path.encode()).hexdigest()[:8]}"
        quarantine_path = self.quarantine_dir / f"{quarantine_id}.json"

        quarantine_entry = {
            "repo_id": repo_id,
            "file_path_hash": sha256(file_path.encode()).hexdigest(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "retry_count": retry_count,
            "max_retries": self.max_retries,
            "quarantined_at": datetime.utcnow().isoformat(),
            "content_size": len(content),
        }

        quarantine_path.write_text(json.dumps(quarantine_entry, indent=2))

        # Store content separately
        content_path = self.quarantine_dir / f"{quarantine_id}.bin"
        content_path.write_bytes(content)

        logger.warning(
            f"File quarantined: {quarantine_id} "
            f"(retry {retry_count}/{self.max_retries})"
        )

    async def retry_quarantined_files(self):
        """Attempt to retry quarantined files."""

        for quarantine_file in self.quarantine_dir.glob("*.json"):
            entry = json.loads(quarantine_file.read_text())

            if entry["retry_count"] < self.max_retries:
                # Attempt retry
                # ... (implementation)
                pass
```

## Health Monitoring

### Connector Health Check
```python
class GitHubConnectorHealth:
    """Health monitoring for GitHub connector."""

    def __init__(self, connector: GitHubRepositoryConnector):
        self.connector = connector

    async def check_health(self) -> Dict[str, Any]:
        """Check connector health status."""

        health = {
            "status": "healthy",
            "checks": {},
            "metrics": {},
        }

        # Check API connectivity
        try:
            # Test API with rate limit query
            rate_limit_ok = await self._check_api_connectivity()
            health["checks"]["api_connectivity"] = "pass" if rate_limit_ok else "fail"
        except Exception as e:
            health["checks"]["api_connectivity"] = "fail"
            health["status"] = "degraded"

        # Check credentials
        try:
            creds_valid = self._check_credentials()
            health["checks"]["credentials"] = "pass" if creds_valid else "fail"
        except Exception:
            health["checks"]["credentials"] = "fail"
            health["status"] = "degraded"

        # Collect metrics
        health["metrics"] = {
            "registered_repositories": len(self.connector.list_repositories()),
            "active_sync_jobs": self._count_active_jobs(),
            "quarantined_files": self._count_quarantined_files(),
        }

        return health
```

## Acceptance Criteria

- ✅ Connector registered with IngestionOrchestrator
- ✅ Scheduled sync jobs execute on interval
- ✅ Webhook events trigger immediate syncs
- ✅ ElementSink receives processed elements
- ✅ StateStore persists sync state reliably
- ✅ Quarantine workflow handles failed files
- ✅ Retry policy works for transient failures
- ✅ Health check reports connector status
- ✅ Job priority and resource limits respected
- ✅ Metrics collected for monitoring

## Test Plan

### Unit Tests
- Job creation and scheduling
- State persistence and loading
- Element sink delivery
- Quarantine entry creation

### Integration Tests
- End-to-end orchestrator registration
- Scheduled job execution
- Webhook event processing
- State store round-trip
- Quarantine and retry workflow

### System Tests
- Multi-repository concurrent sync
- Long-running sync operations
- Orchestrator restart recovery
- Resource limit enforcement

## Implementation Notes

### Job Scheduling with APScheduler
```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

# Schedule repository sync
scheduler.add_job(
    connector.sync_repository,
    trigger="cron",
    minute="*/5",  # Every 5 minutes
    args=[repo_id],
    id=f"github_sync_{repo_id}",
)
```

### Priority Queue for Jobs
```python
class PriorityJobQueue:
    """Priority-based job queue."""

    def __init__(self):
        self.queue = asyncio.PriorityQueue()

    async def add_job(self, job: GitHubConnectorJob):
        """Add job to queue."""
        await self.queue.put((job.priority, job))

    async def get_job(self) -> GitHubConnectorJob:
        """Get highest priority job."""
        _, job = await self.queue.get()
        return job
```

## Open Questions

- Should we support job chaining (sync → issue fetch → PR fetch)?
- How to handle concurrent syncs of the same repository?
- Should we implement backpressure when element sink is slow?
- How to prioritize webhook events vs scheduled syncs?

## Dependencies
- IngestionOrchestrator for job management
- APScheduler for job scheduling
- ElementSink for PKG/vector store delivery
- StateStore for persistence
- Quarantine utilities


