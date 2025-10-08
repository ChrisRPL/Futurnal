"""Unit tests for retry policy system."""

from datetime import datetime
from pathlib import Path

import pytest

from futurnal.orchestrator.models import JobType
from futurnal.orchestrator.retry_policy import (
    FailureType,
    RetryBudget,
    RetryPolicy,
    RetryPolicyRegistry,
    RetryStrategy,
)


class TestRetryStrategy:
    """Test retry strategy enum."""

    def test_all_strategies_defined(self):
        """Test that all expected strategies are defined."""
        assert RetryStrategy.EXPONENTIAL_BACKOFF == "exponential_backoff"
        assert RetryStrategy.LINEAR_BACKOFF == "linear_backoff"
        assert RetryStrategy.FIXED_DELAY == "fixed_delay"
        assert RetryStrategy.IMMEDIATE == "immediate"
        assert RetryStrategy.NO_RETRY == "no_retry"


class TestFailureType:
    """Test failure type enum."""

    def test_all_failure_types_defined(self):
        """Test that all expected failure types are defined."""
        assert FailureType.TRANSIENT == "transient"
        assert FailureType.RATE_LIMITED == "rate_limited"
        assert FailureType.PERMANENT == "permanent"
        assert FailureType.UNKNOWN == "unknown"


class TestRetryPolicyValidation:
    """Test retry policy validation."""

    def test_valid_policy_creation(self):
        """Test creating a valid retry policy."""
        policy = RetryPolicy(
            connector_type=JobType.LOCAL_FILES,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_attempts=5,
            base_delay_seconds=60,
            max_delay_seconds=1800,
        )
        assert policy.connector_type == JobType.LOCAL_FILES
        assert policy.max_attempts == 5

    def test_max_attempts_validation(self):
        """Test that max_attempts is validated correctly."""
        # Valid range: 1-10
        with pytest.raises(ValueError):
            RetryPolicy(
                connector_type=JobType.LOCAL_FILES, max_attempts=0
            )

        with pytest.raises(ValueError):
            RetryPolicy(
                connector_type=JobType.LOCAL_FILES, max_attempts=11
            )

    def test_base_delay_validation(self):
        """Test that base_delay_seconds is validated correctly."""
        # Valid range: 1-3600
        with pytest.raises(ValueError):
            RetryPolicy(
                connector_type=JobType.LOCAL_FILES, base_delay_seconds=0
            )

        with pytest.raises(ValueError):
            RetryPolicy(
                connector_type=JobType.LOCAL_FILES, base_delay_seconds=3601
            )

    def test_jitter_factor_validation(self):
        """Test that jitter_factor is validated correctly."""
        # Valid range: 0.0-1.0
        with pytest.raises(ValueError):
            RetryPolicy(
                connector_type=JobType.LOCAL_FILES, jitter_factor=-0.1
            )

        with pytest.raises(ValueError):
            RetryPolicy(
                connector_type=JobType.LOCAL_FILES, jitter_factor=1.1
            )

    def test_backoff_multiplier_validation(self):
        """Test that backoff_multiplier is validated correctly."""
        # Valid range: 1.0-10.0
        with pytest.raises(ValueError):
            RetryPolicy(
                connector_type=JobType.LOCAL_FILES, backoff_multiplier=0.9
            )

        with pytest.raises(ValueError):
            RetryPolicy(
                connector_type=JobType.LOCAL_FILES, backoff_multiplier=10.1
            )


class TestRetryPolicyDelayCalculation:
    """Test retry delay calculation with various strategies."""

    def test_exponential_backoff_calculation(self):
        """Test exponential backoff delay calculation."""
        policy = RetryPolicy(
            connector_type=JobType.LOCAL_FILES,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay_seconds=60,
            backoff_multiplier=2.0,
            jitter_factor=0.0,  # No jitter for predictable testing
        )

        # Attempt 0: 60 * (2^0) = 60
        assert policy.calculate_delay(0) == 60
        # Attempt 1: 60 * (2^1) = 120
        assert policy.calculate_delay(1) == 120
        # Attempt 2: 60 * (2^2) = 240
        assert policy.calculate_delay(2) == 240
        # Attempt 3: 60 * (2^3) = 480
        assert policy.calculate_delay(3) == 480

    def test_linear_backoff_calculation(self):
        """Test linear backoff delay calculation."""
        policy = RetryPolicy(
            connector_type=JobType.LOCAL_FILES,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            base_delay_seconds=60,
            jitter_factor=0.0,
        )

        # Attempt 0: 60 * (0 + 1) = 60
        assert policy.calculate_delay(0) == 60
        # Attempt 1: 60 * (1 + 1) = 120
        assert policy.calculate_delay(1) == 120
        # Attempt 2: 60 * (2 + 1) = 180
        assert policy.calculate_delay(2) == 180

    def test_fixed_delay_calculation(self):
        """Test fixed delay calculation."""
        policy = RetryPolicy(
            connector_type=JobType.LOCAL_FILES,
            strategy=RetryStrategy.FIXED_DELAY,
            base_delay_seconds=60,
            jitter_factor=0.0,
        )

        assert policy.calculate_delay(0) == 60
        assert policy.calculate_delay(1) == 60
        assert policy.calculate_delay(2) == 60
        assert policy.calculate_delay(5) == 60

    def test_immediate_strategy(self):
        """Test immediate strategy returns zero delay."""
        policy = RetryPolicy(
            connector_type=JobType.LOCAL_FILES,
            strategy=RetryStrategy.IMMEDIATE,
        )

        assert policy.calculate_delay(0) == 0
        assert policy.calculate_delay(1) == 0

    def test_no_retry_strategy(self):
        """Test no_retry strategy returns zero delay."""
        policy = RetryPolicy(
            connector_type=JobType.LOCAL_FILES,
            strategy=RetryStrategy.NO_RETRY,
        )

        assert policy.calculate_delay(0) == 0

    def test_delay_capped_at_max(self):
        """Test that delay is capped at max_delay_seconds."""
        policy = RetryPolicy(
            connector_type=JobType.LOCAL_FILES,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay_seconds=60,
            max_delay_seconds=200,
            backoff_multiplier=2.0,
            jitter_factor=0.0,
        )

        # Attempt 0: 60 (within limit)
        assert policy.calculate_delay(0) == 60
        # Attempt 1: 120 (within limit)
        assert policy.calculate_delay(1) == 120
        # Attempt 2: would be 240, but capped at 200
        assert policy.calculate_delay(2) == 200
        # Attempt 3: would be 480, but capped at 200
        assert policy.calculate_delay(3) == 200

    def test_rate_limit_delay_override(self):
        """Test that rate-limited failures use override delay."""
        policy = RetryPolicy(
            connector_type=JobType.GITHUB_REPOSITORY,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay_seconds=60,
            rate_limit_delay_seconds=900,
            jitter_factor=0.0,
        )

        # Normal failure: use base delay
        assert policy.calculate_delay(0, FailureType.TRANSIENT) == 60

        # Rate-limited failure: use override delay
        delay = policy.calculate_delay(0, FailureType.RATE_LIMITED)
        assert delay == 900

    def test_jitter_within_bounds(self):
        """Test that jitter stays within expected bounds."""
        policy = RetryPolicy(
            connector_type=JobType.LOCAL_FILES,
            strategy=RetryStrategy.FIXED_DELAY,
            base_delay_seconds=100,
            jitter_factor=0.2,  # ±20%
        )

        # Run multiple iterations to test randomness
        delays = [policy.calculate_delay(0) for _ in range(100)]

        # All delays should be within ±20% of base (80-120)
        assert all(80 <= d <= 120 for d in delays)

        # Should have some variation (not all the same)
        assert len(set(delays)) > 1

    def test_jitter_distribution(self):
        """Test that jitter produces reasonable distribution."""
        policy = RetryPolicy(
            connector_type=JobType.LOCAL_FILES,
            strategy=RetryStrategy.FIXED_DELAY,
            base_delay_seconds=100,
            jitter_factor=0.3,  # ±30%
        )

        delays = [policy.calculate_delay(0) for _ in range(1000)]

        # Mean should be close to base delay
        mean_delay = sum(delays) / len(delays)
        assert 95 <= mean_delay <= 105  # Within 5% of expected

        # Should have good spread
        min_delay = min(delays)
        max_delay = max(delays)
        assert min_delay < 80  # Some values in lower range
        assert max_delay > 120  # Some values in upper range


class TestRetryBudget:
    """Test retry budget tracking."""

    def test_budget_creation(self):
        """Test creating a retry budget."""
        budget = RetryBudget(
            job_id="test-job",
            job_type=JobType.LOCAL_FILES,
        )

        assert budget.job_id == "test-job"
        assert budget.job_type == JobType.LOCAL_FILES
        assert budget.attempts == 0
        assert budget.failure_type == FailureType.UNKNOWN

    def test_can_retry_within_limits(self):
        """Test that retry is allowed within budget."""
        policy = RetryPolicy(
            connector_type=JobType.LOCAL_FILES,
            max_attempts=3,
        )

        budget = RetryBudget(
            job_id="test-job",
            job_type=JobType.LOCAL_FILES,
            attempts=0,
        )

        assert budget.can_retry(policy) is True

        budget.attempts = 1
        assert budget.can_retry(policy) is True

        budget.attempts = 2
        assert budget.can_retry(policy) is True

    def test_can_retry_exhausted(self):
        """Test that retry is denied when budget exhausted."""
        policy = RetryPolicy(
            connector_type=JobType.LOCAL_FILES,
            max_attempts=3,
        )

        budget = RetryBudget(
            job_id="test-job",
            job_type=JobType.LOCAL_FILES,
            attempts=3,
        )

        assert budget.can_retry(policy) is False

    def test_permanent_failure_no_retry(self):
        """Test that permanent failures skip retry."""
        policy = RetryPolicy(
            connector_type=JobType.LOCAL_FILES,
            max_attempts=5,
            permanent_failures_no_retry=True,
        )

        budget = RetryBudget(
            job_id="test-job",
            job_type=JobType.LOCAL_FILES,
            attempts=0,
            failure_type=FailureType.PERMANENT,
        )

        assert budget.can_retry(policy) is False

    def test_transient_max_attempts_override(self):
        """Test that transient failures use override max attempts."""
        policy = RetryPolicy(
            connector_type=JobType.IMAP_MAILBOX,
            max_attempts=5,
            transient_max_attempts=7,
        )

        # Normal failure: limited to 5 attempts
        budget_normal = RetryBudget(
            job_id="test-job",
            job_type=JobType.IMAP_MAILBOX,
            attempts=5,
            failure_type=FailureType.UNKNOWN,
        )
        assert budget_normal.can_retry(policy) is False

        # Transient failure: allowed up to 7 attempts
        budget_transient = RetryBudget(
            job_id="test-job",
            job_type=JobType.IMAP_MAILBOX,
            attempts=5,
            failure_type=FailureType.TRANSIENT,
        )
        assert budget_transient.can_retry(policy) is True

        budget_transient.attempts = 7
        assert budget_transient.can_retry(policy) is False

    def test_next_delay_calculation(self):
        """Test next_delay delegates to policy correctly."""
        policy = RetryPolicy(
            connector_type=JobType.LOCAL_FILES,
            strategy=RetryStrategy.FIXED_DELAY,
            base_delay_seconds=100,
            jitter_factor=0.0,
        )

        budget = RetryBudget(
            job_id="test-job",
            job_type=JobType.LOCAL_FILES,
            attempts=0,
            failure_type=FailureType.TRANSIENT,
        )

        assert budget.next_delay(policy) == 100


class TestRetryPolicyRegistry:
    """Test retry policy registry."""

    def test_default_policies_all_connectors(self):
        """Test that default policies exist for all connector types."""
        registry = RetryPolicyRegistry()

        # All connector types should have policies
        assert JobType.LOCAL_FILES in registry.list_policies()
        assert JobType.OBSIDIAN_VAULT in registry.list_policies()
        assert JobType.IMAP_MAILBOX in registry.list_policies()
        assert JobType.GITHUB_REPOSITORY in registry.list_policies()

    def test_get_policy_existing(self):
        """Test getting an existing policy."""
        registry = RetryPolicyRegistry()
        policy = registry.get_policy(JobType.LOCAL_FILES)

        assert policy is not None
        assert policy.connector_type == JobType.LOCAL_FILES

    def test_set_policy_override(self):
        """Test overriding a policy."""
        registry = RetryPolicyRegistry()

        custom_policy = RetryPolicy(
            connector_type=JobType.LOCAL_FILES,
            max_attempts=10,
            base_delay_seconds=5,
        )

        registry.set_policy(custom_policy)
        retrieved_policy = registry.get_policy(JobType.LOCAL_FILES)

        assert retrieved_policy.max_attempts == 10
        assert retrieved_policy.base_delay_seconds == 5

    def test_default_policy_characteristics(self):
        """Test that default policies have expected characteristics."""
        registry = RetryPolicyRegistry()

        # Local files: fast retry
        local_policy = registry.get_policy(JobType.LOCAL_FILES)
        assert local_policy.base_delay_seconds == 30
        assert local_policy.max_attempts == 3

        # IMAP: conservative retry with more attempts
        imap_policy = registry.get_policy(JobType.IMAP_MAILBOX)
        assert imap_policy.base_delay_seconds == 120
        assert imap_policy.max_attempts == 5
        assert imap_policy.transient_max_attempts == 7
        assert imap_policy.rate_limit_delay_seconds == 600

        # GitHub: rate-limit aware
        github_policy = registry.get_policy(JobType.GITHUB_REPOSITORY)
        assert github_policy.base_delay_seconds == 300
        assert github_policy.rate_limit_delay_seconds == 900


class TestRetryPolicyConfiguration:
    """Test YAML configuration loading."""

    def test_load_valid_yaml_config(self, tmp_path: Path):
        """Test loading valid YAML configuration."""
        config_path = tmp_path / "retry_policies.yaml"
        config_path.write_text(
            """
retry_policies:
  local_files:
    strategy: exponential_backoff
    max_attempts: 5
    base_delay_seconds: 10
    max_delay_seconds: 100
    jitter_factor: 0.1
    backoff_multiplier: 3.0
"""
        )

        registry = RetryPolicyRegistry(config_path=config_path)
        policy = registry.get_policy(JobType.LOCAL_FILES)

        assert policy.max_attempts == 5
        assert policy.base_delay_seconds == 10
        assert policy.max_delay_seconds == 100
        assert policy.jitter_factor == 0.1
        assert policy.backoff_multiplier == 3.0

    def test_load_config_with_overrides(self, tmp_path: Path):
        """Test loading config with failure-type overrides."""
        config_path = tmp_path / "retry_policies.yaml"
        config_path.write_text(
            """
retry_policies:
  imap_mailbox:
    strategy: exponential_backoff
    max_attempts: 3
    base_delay_seconds: 60
    transient_max_attempts: 10
    rate_limit_delay_seconds: 1200
    permanent_failures_no_retry: false
"""
        )

        registry = RetryPolicyRegistry(config_path=config_path)
        policy = registry.get_policy(JobType.IMAP_MAILBOX)

        assert policy.transient_max_attempts == 10
        assert policy.rate_limit_delay_seconds == 1200
        assert policy.permanent_failures_no_retry is False

    def test_load_config_invalid_yaml(self, tmp_path: Path):
        """Test that invalid YAML raises error."""
        config_path = tmp_path / "retry_policies.yaml"
        config_path.write_text("invalid: yaml: structure: [")

        with pytest.raises(ValueError, match="Invalid YAML"):
            RetryPolicyRegistry(config_path=config_path)

    def test_load_config_missing_key(self, tmp_path: Path):
        """Test that missing 'retry_policies' key raises error."""
        config_path = tmp_path / "retry_policies.yaml"
        config_path.write_text(
            """
wrong_key:
  local_files:
    max_attempts: 5
"""
        )

        with pytest.raises(ValueError, match="retry_policies"):
            RetryPolicyRegistry(config_path=config_path)

    def test_load_config_unknown_connector(self, tmp_path: Path):
        """Test that unknown connector type raises error."""
        config_path = tmp_path / "retry_policies.yaml"
        config_path.write_text(
            """
retry_policies:
  unknown_connector:
    max_attempts: 5
"""
        )

        with pytest.raises(ValueError, match="Unknown connector type"):
            RetryPolicyRegistry(config_path=config_path)

    def test_load_config_validation_error(self, tmp_path: Path):
        """Test that invalid parameter values raise validation error."""
        config_path = tmp_path / "retry_policies.yaml"
        config_path.write_text(
            """
retry_policies:
  local_files:
    max_attempts: 20  # Exceeds max of 10
"""
        )

        with pytest.raises(ValueError):
            RetryPolicyRegistry(config_path=config_path)

    def test_nonexistent_config_uses_defaults(self, tmp_path: Path):
        """Test that nonexistent config file uses defaults."""
        config_path = tmp_path / "nonexistent.yaml"
        registry = RetryPolicyRegistry(config_path=config_path)

        # Should use default policies
        policy = registry.get_policy(JobType.LOCAL_FILES)
        assert policy.max_attempts == 3  # Default value


class TestRetryPolicyPerformance:
    """Test performance characteristics of retry policy system."""

    def test_jitter_calculation_performance(self):
        """Test that jitter calculation is fast."""
        import time

        policy = RetryPolicy(
            connector_type=JobType.LOCAL_FILES,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter_factor=0.3,
        )

        start = time.perf_counter()
        for _ in range(1000):
            policy.calculate_delay(2)
        duration = time.perf_counter() - start

        # Should complete 1000 calculations in under 100ms
        assert duration < 0.1

    def test_policy_registry_lookup_performance(self):
        """Test that policy lookups are fast."""
        import time

        registry = RetryPolicyRegistry()

        start = time.perf_counter()
        for _ in range(10000):
            registry.get_policy(JobType.LOCAL_FILES)
        duration = time.perf_counter() - start

        # Should complete 10000 lookups in under 100ms
        assert duration < 0.1
