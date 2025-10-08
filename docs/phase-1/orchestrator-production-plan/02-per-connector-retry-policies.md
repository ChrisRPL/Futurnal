Summary: Implement configurable retry policies per connector with failure-specific strategies and comprehensive retry telemetry.

# 02 · Per-Connector Retry Policies

## Purpose
Enable fine-grained retry configuration per connector type, allowing different retry strategies for local files (fast retry), email sync (conservative retry), and GitHub repositories (rate-limit aware retry). Ensures the Ghost's experiential learning pipeline adapts retry behavior to the unique characteristics and failure modes of each data source.

## Scope
- RetryPolicy configuration schema per connector
- Failure-type-specific retry strategies (transient vs. permanent failures)
- Exponential backoff with configurable jitter
- Per-connector retry budget tracking
- Integration with existing retry mechanism in IngestionOrchestrator
- Retry telemetry per connector type
- Configuration validation and sensible defaults

## Requirements Alignment
- **Configurability**: Connector-specific retry policies as specified in feature requirements
- **Visibility**: Retry telemetry provides insight into failure patterns per connector
- **Fault Tolerance**: Adaptive retry strategies improve recovery success rates
- **Performance**: Avoid wasting resources on unrecoverable errors
- **Observability**: Track retry budgets and success rates per connector

## Data Model

### RetryPolicy Schema
```python
class RetryStrategy(str, Enum):
    """Retry strategy types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Standard: 60s, 120s, 240s...
    LINEAR_BACKOFF = "linear_backoff"            # Linear: 60s, 120s, 180s...
    FIXED_DELAY = "fixed_delay"                  # Constant: 60s, 60s, 60s...
    IMMEDIATE = "immediate"                      # No delay (testing only)
    NO_RETRY = "no_retry"                        # Fail immediately

class FailureType(str, Enum):
    """Failure classification for retry decisions."""
    TRANSIENT = "transient"          # Network hiccups, temporary resource limits
    RATE_LIMITED = "rate_limited"    # API rate limits (need longer backoff)
    PERMANENT = "permanent"          # Permission denied, invalid credentials
    UNKNOWN = "unknown"              # Unclassified errors

class RetryPolicy(BaseModel):
    """Configurable retry policy for a connector."""
    connector_type: JobType                      # Which connector this applies to
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_attempts: int = Field(default=3, ge=1, le=10)
    base_delay_seconds: int = Field(default=60, ge=1, le=3600)
    max_delay_seconds: int = Field(default=3600, ge=1, le=86400)
    jitter_factor: float = Field(default=0.2, ge=0.0, le=1.0)  # 20% jitter
    backoff_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)

    # Failure-specific overrides
    transient_max_attempts: Optional[int] = None
    rate_limit_delay_seconds: Optional[int] = None
    permanent_failures_no_retry: bool = True

    def calculate_delay(
        self,
        attempt: int,
        failure_type: FailureType = FailureType.UNKNOWN,
    ) -> int:
        """Calculate retry delay with jitter."""
```

### RetryPolicyRegistry
```python
class RetryPolicyRegistry:
    """Manages retry policies per connector type."""

    # Default policies per connector
    DEFAULT_POLICIES: Dict[JobType, RetryPolicy] = {
        JobType.LOCAL_FILES: RetryPolicy(
            connector_type=JobType.LOCAL_FILES,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_attempts=3,
            base_delay_seconds=30,  # Fast retry for local files
            max_delay_seconds=300,
        ),
        JobType.OBSIDIAN_VAULT: RetryPolicy(
            connector_type=JobType.OBSIDIAN_VAULT,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_attempts=3,
            base_delay_seconds=30,
            max_delay_seconds=300,
        ),
        JobType.IMAP_MAILBOX: RetryPolicy(
            connector_type=JobType.IMAP_MAILBOX,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_attempts=5,  # More attempts for email sync
            base_delay_seconds=120,  # Conservative for network
            max_delay_seconds=1800,
            rate_limit_delay_seconds=600,  # 10 min for rate limits
        ),
        JobType.GITHUB_REPOSITORY: RetryPolicy(
            connector_type=JobType.GITHUB_REPOSITORY,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_attempts=5,
            base_delay_seconds=300,  # 5 min base for API limits
            max_delay_seconds=3600,  # 1 hour max
            rate_limit_delay_seconds=900,  # 15 min for rate limits
        ),
    }

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize with optional custom configuration."""
        self._policies = self.DEFAULT_POLICIES.copy()
        if config_path and config_path.exists():
            self._load_config(config_path)

    def get_policy(self, job_type: JobType) -> RetryPolicy:
        """Get retry policy for connector type."""
        return self._policies.get(job_type, self._create_default_policy(job_type))

    def set_policy(self, policy: RetryPolicy) -> None:
        """Override policy for connector type."""
        self._policies[policy.connector_type] = policy

    def _load_config(self, config_path: Path) -> None:
        """Load custom retry policies from YAML/JSON."""
```

### RetryBudget Tracking
```python
@dataclass
class RetryBudget:
    """Tracks retry attempts for a specific job."""
    job_id: str
    job_type: JobType
    attempts: int = 0
    failure_type: FailureType = FailureType.UNKNOWN
    first_attempt_at: Optional[datetime] = None
    last_attempt_at: Optional[datetime] = None
    total_delay_seconds: int = 0

    def can_retry(self, policy: RetryPolicy) -> bool:
        """Check if retry budget allows another attempt."""
        max_attempts = policy.max_attempts

        # Override for specific failure types
        if self.failure_type == FailureType.TRANSIENT and policy.transient_max_attempts:
            max_attempts = policy.transient_max_attempts
        elif self.failure_type == FailureType.PERMANENT and policy.permanent_failures_no_retry:
            return False

        return self.attempts < max_attempts

    def next_delay(self, policy: RetryPolicy) -> int:
        """Calculate next retry delay based on policy."""
        return policy.calculate_delay(self.attempts, self.failure_type)
```

## Component Design

### Integration with IngestionOrchestrator
```python
class IngestionOrchestrator:
    def __init__(
        self,
        *,
        job_queue: JobQueue,
        retry_policy_registry: Optional[RetryPolicyRegistry] = None,
        # ... existing params
    ) -> None:
        self._retry_policies = retry_policy_registry or RetryPolicyRegistry()
        self._retry_budgets: Dict[str, RetryBudget] = {}
        # ... existing initialization

    async def _maybe_retry(self, job: IngestionJob) -> None:
        """Enhanced retry with per-connector policies."""
        # Get or create retry budget
        budget = self._retry_budgets.get(job.job_id)
        if not budget:
            budget = RetryBudget(
                job_id=job.job_id,
                job_type=job.job_type,
                first_attempt_at=datetime.utcnow(),
            )
            self._retry_budgets[job.job_id] = budget

        # Classify failure type
        error_message = job.payload.get("error", "")
        budget.failure_type = self._classify_failure_type(error_message)

        # Get connector-specific policy
        policy = self._retry_policies.get_policy(job.job_type)

        # Check if retry allowed
        if not budget.can_retry(policy):
            logger.info(
                "Retry budget exhausted, quarantining job",
                extra={
                    "ingestion_job_id": job.job_id,
                    "job_type": job.job_type.value,
                    "attempts": budget.attempts,
                    "max_attempts": policy.max_attempts,
                },
            )
            await self._quarantine_job(job)
            del self._retry_budgets[job.job_id]
            return

        # Calculate delay
        delay_seconds = budget.next_delay(policy)
        budget.attempts += 1
        budget.last_attempt_at = datetime.utcnow()
        budget.total_delay_seconds += delay_seconds

        logger.info(
            "Scheduling retry",
            extra={
                "ingestion_job_id": job.job_id,
                "job_type": job.job_type.value,
                "attempt": budget.attempts,
                "max_attempts": policy.max_attempts,
                "delay_seconds": delay_seconds,
                "failure_type": budget.failure_type.value,
            },
        )

        # Update job payload
        job.payload["attempts"] = budget.attempts
        job.payload["failure_type"] = budget.failure_type.value
        job.payload["trigger"] = "retry"

        # Reschedule with calculated delay
        self._job_queue.reschedule(job.job_id, delay_seconds)

        # Record telemetry
        if self._telemetry:
            self._telemetry.record(
                job_id=job.job_id,
                duration=0.0,
                status="retry_scheduled",
                metadata={
                    "job_type": job.job_type.value,
                    "attempt": budget.attempts,
                    "delay_seconds": delay_seconds,
                    "failure_type": budget.failure_type.value,
                },
            )

    def _classify_failure_type(self, error_message: str) -> FailureType:
        """Classify failure for retry strategy selection."""
        error_lower = error_message.lower()

        # Permanent failures
        if any(term in error_lower for term in [
            "permission denied",
            "access denied",
            "authentication failed",
            "invalid credentials",
            "not found",
        ]):
            return FailureType.PERMANENT

        # Rate limited
        if any(term in error_lower for term in [
            "rate limit",
            "too many requests",
            "quota exceeded",
            "429",
        ]):
            return FailureType.RATE_LIMITED

        # Transient failures
        if any(term in error_lower for term in [
            "timeout",
            "connection refused",
            "temporary",
            "unavailable",
            "network",
            "503",
            "502",
        ]):
            return FailureType.TRANSIENT

        return FailureType.UNKNOWN
```

### RetryPolicy Implementation
```python
def calculate_delay(
    self,
    attempt: int,
    failure_type: FailureType = FailureType.UNKNOWN,
) -> int:
    """Calculate retry delay with jitter."""
    import random

    # Handle rate limits specially
    if failure_type == FailureType.RATE_LIMITED and self.rate_limit_delay_seconds:
        base_delay = self.rate_limit_delay_seconds
    else:
        base_delay = self.base_delay_seconds

    # Calculate delay based on strategy
    if self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
        delay = base_delay * (self.backoff_multiplier ** attempt)
    elif self.strategy == RetryStrategy.LINEAR_BACKOFF:
        delay = base_delay * (attempt + 1)
    elif self.strategy == RetryStrategy.FIXED_DELAY:
        delay = base_delay
    elif self.strategy == RetryStrategy.IMMEDIATE:
        delay = 0
    else:  # NO_RETRY
        delay = 0

    # Cap at max delay
    delay = min(delay, self.max_delay_seconds)

    # Add jitter to avoid thundering herd
    if self.jitter_factor > 0:
        jitter_amount = delay * self.jitter_factor
        jitter = random.uniform(-jitter_amount, jitter_amount)
        delay = max(1, int(delay + jitter))

    return int(delay)
```

## Configuration Format

### YAML Configuration
```yaml
# ~/.futurnal/config/retry_policies.yaml
retry_policies:
  local_files:
    strategy: exponential_backoff
    max_attempts: 3
    base_delay_seconds: 30
    max_delay_seconds: 300
    jitter_factor: 0.2
    backoff_multiplier: 2.0
    permanent_failures_no_retry: true

  imap_mailbox:
    strategy: exponential_backoff
    max_attempts: 5
    base_delay_seconds: 120
    max_delay_seconds: 1800
    jitter_factor: 0.3
    backoff_multiplier: 2.0
    transient_max_attempts: 7
    rate_limit_delay_seconds: 600

  github_repository:
    strategy: exponential_backoff
    max_attempts: 5
    base_delay_seconds: 300
    max_delay_seconds: 3600
    jitter_factor: 0.25
    backoff_multiplier: 2.0
    rate_limit_delay_seconds: 900
    permanent_failures_no_retry: true
```

## Acceptance Criteria

- ✅ RetryPolicy schema supports all configuration parameters
- ✅ RetryPolicyRegistry loads default policies per connector type
- ✅ RetryPolicyRegistry loads custom policies from YAML configuration
- ✅ RetryBudget tracks attempts and failure types per job
- ✅ Failure classification correctly identifies transient/permanent/rate-limited errors
- ✅ Exponential backoff with jitter prevents thundering herd
- ✅ Rate-limited failures use extended delay periods
- ✅ Permanent failures skip retry and go straight to quarantine
- ✅ Retry telemetry captures per-connector retry patterns
- ✅ Configuration validation rejects invalid retry policies
- ✅ CLI command to display current retry policies per connector
- ✅ Documentation explains retry policy tuning for each connector type

## Test Plan

### Unit Tests
- `test_retry_policy_calculation.py`: Delay calculation with various strategies
- `test_jitter_distribution.py`: Verify jitter is within expected bounds
- `test_failure_classification.py`: Transient/permanent/rate-limited detection
- `test_retry_budget_exhaustion.py`: Budget limits respected
- `test_policy_registry_defaults.py`: Default policies per connector
- `test_policy_configuration_loading.py`: YAML config parsing

### Integration Tests
- `test_connector_specific_retries.py`: Different policies per connector
- `test_rate_limit_retry.py`: Extended delays for rate-limited failures
- `test_permanent_failure_no_retry.py`: Skip retry for permanent errors
- `test_retry_telemetry.py`: Telemetry captures retry metrics
- `test_retry_budget_cleanup.py`: Budgets cleaned after quarantine

### Performance Tests
- `test_jitter_overhead.py`: Jitter calculation performance
- `test_concurrent_retry_scheduling.py`: Thread-safe budget tracking

### Validation Tests
- `test_invalid_retry_policy_rejection.py`: Invalid config rejected
- `test_retry_policy_bounds_checking.py`: Parameter constraints enforced

## Implementation Notes

### Exponential Backoff Formula
```python
delay = base_delay * (multiplier ** attempt) + jitter
```

Examples with base=60s, multiplier=2.0, jitter=0.2:
- Attempt 0: 60s ± 12s = 48-72s
- Attempt 1: 120s ± 24s = 96-144s
- Attempt 2: 240s ± 48s = 192-288s

### Jitter Benefits
- Prevents thundering herd when multiple jobs fail simultaneously
- Distributes retry load over time
- Reduces contention on shared resources

### Failure Type Detection
Enhanced detection using exception types:
```python
def _classify_failure_type(
    self,
    error_message: str,
    exception: Optional[Exception] = None,
) -> FailureType:
    """Enhanced classification using exception instance."""
    if exception:
        # Permission errors are permanent
        if isinstance(exception, PermissionError):
            return FailureType.PERMANENT

        # Timeouts are transient
        if isinstance(exception, (TimeoutError, asyncio.TimeoutError)):
            return FailureType.TRANSIENT

        # HTTP status codes
        if hasattr(exception, "status_code"):
            if exception.status_code == 429:
                return FailureType.RATE_LIMITED
            elif exception.status_code in [502, 503, 504]:
                return FailureType.TRANSIENT
            elif exception.status_code in [401, 403, 404]:
                return FailureType.PERMANENT

    # Fallback to message pattern matching
    return self._classify_by_message(error_message)
```

### Telemetry Schema
```json
{
  "retry_metrics": {
    "local_files": {
      "total_retries": 42,
      "successful_retries": 35,
      "failed_retries": 7,
      "avg_attempts_until_success": 1.8,
      "by_failure_type": {
        "transient": 30,
        "permanent": 5,
        "rate_limited": 0,
        "unknown": 7
      }
    },
    "imap_mailbox": {
      "total_retries": 18,
      "successful_retries": 15,
      "failed_retries": 3,
      "avg_attempts_until_success": 2.3,
      "by_failure_type": {
        "transient": 12,
        "rate_limited": 3,
        "permanent": 1,
        "unknown": 2
      }
    }
  }
}
```

## Open Questions

- Should retry policies be configurable per source (not just connector type)?
- How to handle adaptive retry (learn optimal delays from past successes)?
- Should we support custom retry strategies via plugin system?
- What's the appropriate jitter factor for each connector type?
- Should retry budget persist across orchestrator restarts?
- How to expose retry configuration in the operator CLI?
- Should we provide retry policy templates for common scenarios?

## Dependencies

- Existing IngestionOrchestrator retry mechanism
- JobQueue for reschedule operations
- TelemetryRecorder for retry metrics
- Configuration management (YAML/JSON parsing)
- QuarantineStore (Task 01) for failed job handling


