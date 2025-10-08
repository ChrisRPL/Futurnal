"""Configurable retry policies per connector with failure-specific strategies.

Implements exponential backoff with jitter, per-connector retry budgets,
and failure-type-specific retry strategies for resilient ingestion.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import yaml
from pydantic import BaseModel, Field

from .models import JobType


class RetryStrategy(str, Enum):
    """Retry strategy types for different backoff patterns."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Standard: 60s, 120s, 240s...
    LINEAR_BACKOFF = "linear_backoff"  # Linear: 60s, 120s, 180s...
    FIXED_DELAY = "fixed_delay"  # Constant: 60s, 60s, 60s...
    IMMEDIATE = "immediate"  # No delay (testing only)
    NO_RETRY = "no_retry"  # Fail immediately


class FailureType(str, Enum):
    """Failure classification for retry decisions."""

    TRANSIENT = "transient"  # Network hiccups, temporary resource limits
    RATE_LIMITED = "rate_limited"  # API rate limits (need longer backoff)
    PERMANENT = "permanent"  # Permission denied, invalid credentials
    UNKNOWN = "unknown"  # Unclassified errors


class RetryPolicy(BaseModel):
    """Configurable retry policy for a connector.

    Defines retry behavior including strategy, max attempts, delays,
    jitter, and failure-type-specific overrides.

    Attributes:
        connector_type: Which connector this policy applies to
        strategy: Backoff strategy to use
        max_attempts: Maximum retry attempts (1-10)
        base_delay_seconds: Base delay in seconds (1-3600)
        max_delay_seconds: Maximum delay cap in seconds (1-86400)
        jitter_factor: Random jitter factor (0.0-1.0, default 0.2)
        backoff_multiplier: Multiplier for exponential backoff (1.0-10.0)
        transient_max_attempts: Override max attempts for transient failures
        rate_limit_delay_seconds: Override delay for rate-limited failures
        permanent_failures_no_retry: Skip retry for permanent failures
    """

    connector_type: JobType
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_attempts: int = Field(default=3, ge=1, le=10)
    base_delay_seconds: int = Field(default=60, ge=1, le=3600)
    max_delay_seconds: int = Field(default=3600, ge=1, le=86400)
    jitter_factor: float = Field(default=0.2, ge=0.0, le=1.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)

    # Failure-specific overrides
    transient_max_attempts: Optional[int] = Field(default=None, ge=1, le=20)
    rate_limit_delay_seconds: Optional[int] = Field(default=None, ge=1, le=7200)
    permanent_failures_no_retry: bool = True

    def calculate_delay(
        self,
        attempt: int,
        failure_type: FailureType = FailureType.UNKNOWN,
    ) -> int:
        """Calculate retry delay with jitter.

        Args:
            attempt: Current retry attempt number (0-indexed)
            failure_type: Type of failure for special handling

        Returns:
            Delay in seconds with jitter applied
        """
        # Handle rate limits specially
        if failure_type == FailureType.RATE_LIMITED and self.rate_limit_delay_seconds:
            base_delay = self.rate_limit_delay_seconds
        else:
            base_delay = self.base_delay_seconds

        # Calculate delay based on strategy
        if self.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = base_delay * (self.backoff_multiplier**attempt)
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
        if self.jitter_factor > 0 and delay > 0:
            jitter_amount = delay * self.jitter_factor
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay = max(1, int(delay + jitter))

        return int(delay)


@dataclass
class RetryBudget:
    """Tracks retry attempts for a specific job.

    Maintains state for a single job's retry lifecycle including
    attempt counts, failure classification, and timing information.

    Attributes:
        job_id: Unique job identifier
        job_type: Connector type for policy lookup
        attempts: Number of retry attempts made
        failure_type: Classification of failure for strategy selection
        first_attempt_at: Timestamp of first failure
        last_attempt_at: Timestamp of most recent retry
        total_delay_seconds: Cumulative delay across all retries
    """

    job_id: str
    job_type: JobType
    attempts: int = 0
    failure_type: FailureType = FailureType.UNKNOWN
    first_attempt_at: Optional[datetime] = None
    last_attempt_at: Optional[datetime] = None
    total_delay_seconds: int = 0

    def can_retry(self, policy: RetryPolicy) -> bool:
        """Check if retry budget allows another attempt.

        Args:
            policy: Retry policy to check against

        Returns:
            True if retry is allowed, False otherwise
        """
        max_attempts = policy.max_attempts

        # Override for specific failure types
        if (
            self.failure_type == FailureType.TRANSIENT
            and policy.transient_max_attempts is not None
        ):
            max_attempts = policy.transient_max_attempts
        elif (
            self.failure_type == FailureType.PERMANENT
            and policy.permanent_failures_no_retry
        ):
            return False

        return self.attempts < max_attempts

    def next_delay(self, policy: RetryPolicy) -> int:
        """Calculate next retry delay based on policy.

        Args:
            policy: Retry policy to use for calculation

        Returns:
            Delay in seconds for next retry
        """
        return policy.calculate_delay(self.attempts, self.failure_type)


class RetryPolicyRegistry:
    """Manages retry policies per connector type.

    Provides default policies for each connector type and supports
    loading custom policies from YAML configuration files.

    Attributes:
        DEFAULT_POLICIES: Default retry policies for each connector
    """

    # Default policies per connector
    DEFAULT_POLICIES: Dict[JobType, RetryPolicy] = {
        JobType.LOCAL_FILES: RetryPolicy(
            connector_type=JobType.LOCAL_FILES,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_attempts=3,
            base_delay_seconds=30,  # Fast retry for local files
            max_delay_seconds=300,
            jitter_factor=0.2,
            backoff_multiplier=2.0,
        ),
        JobType.OBSIDIAN_VAULT: RetryPolicy(
            connector_type=JobType.OBSIDIAN_VAULT,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_attempts=3,
            base_delay_seconds=30,  # Fast retry for local vaults
            max_delay_seconds=300,
            jitter_factor=0.2,
            backoff_multiplier=2.0,
        ),
        JobType.IMAP_MAILBOX: RetryPolicy(
            connector_type=JobType.IMAP_MAILBOX,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_attempts=5,  # More attempts for email sync
            base_delay_seconds=120,  # Conservative for network
            max_delay_seconds=1800,
            jitter_factor=0.3,
            backoff_multiplier=2.0,
            transient_max_attempts=7,  # Extra attempts for transient failures
            rate_limit_delay_seconds=600,  # 10 min for rate limits
        ),
        JobType.GITHUB_REPOSITORY: RetryPolicy(
            connector_type=JobType.GITHUB_REPOSITORY,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_attempts=5,
            base_delay_seconds=300,  # 5 min base for API limits
            max_delay_seconds=3600,  # 1 hour max
            jitter_factor=0.25,
            backoff_multiplier=2.0,
            rate_limit_delay_seconds=900,  # 15 min for rate limits
            permanent_failures_no_retry=True,
        ),
    }

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize with optional custom configuration.

        Args:
            config_path: Optional path to YAML configuration file
        """
        self._policies = self.DEFAULT_POLICIES.copy()
        if config_path and config_path.exists():
            self._load_config(config_path)

    def get_policy(self, job_type: JobType) -> RetryPolicy:
        """Get retry policy for connector type.

        Args:
            job_type: Connector type

        Returns:
            RetryPolicy for the connector (default if not configured)
        """
        return self._policies.get(job_type, self._create_default_policy(job_type))

    def set_policy(self, policy: RetryPolicy) -> None:
        """Override policy for connector type.

        Args:
            policy: New retry policy to register
        """
        self._policies[policy.connector_type] = policy

    def list_policies(self) -> Dict[JobType, RetryPolicy]:
        """List all registered policies.

        Returns:
            Dictionary mapping connector types to policies
        """
        return self._policies.copy()

    def _load_config(self, config_path: Path) -> None:
        """Load custom retry policies from YAML configuration.

        Args:
            config_path: Path to YAML configuration file

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if not isinstance(config, dict) or "retry_policies" not in config:
                raise ValueError("Configuration must contain 'retry_policies' key")

            policies_config = config["retry_policies"]
            if not isinstance(policies_config, dict):
                raise ValueError("'retry_policies' must be a dictionary")

            # Map string keys to JobType enum
            job_type_mapping = {
                "local_files": JobType.LOCAL_FILES,
                "obsidian_vault": JobType.OBSIDIAN_VAULT,
                "imap_mailbox": JobType.IMAP_MAILBOX,
                "github_repository": JobType.GITHUB_REPOSITORY,
            }

            for key, policy_dict in policies_config.items():
                if key not in job_type_mapping:
                    raise ValueError(
                        f"Unknown connector type '{key}'. "
                        f"Valid types: {list(job_type_mapping.keys())}"
                    )

                job_type = job_type_mapping[key]
                policy_dict["connector_type"] = job_type

                # Validate and create policy
                policy = RetryPolicy(**policy_dict)
                self._policies[job_type] = policy

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to load retry policy configuration: {e}") from e

    def _create_default_policy(self, job_type: JobType) -> RetryPolicy:
        """Create a default policy for unknown connector types.

        Args:
            job_type: Connector type

        Returns:
            Default retry policy
        """
        return RetryPolicy(
            connector_type=job_type,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_attempts=3,
            base_delay_seconds=60,
            max_delay_seconds=1800,
            jitter_factor=0.2,
            backoff_multiplier=2.0,
        )
