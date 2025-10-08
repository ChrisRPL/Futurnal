"""Tests for fairness metrics calculations and data models."""

import pytest

from futurnal.orchestrator.load_test import (
    ConnectorLoad,
    ConnectorMetrics,
    FairnessMetrics,
    LoadTestConfig,
    PriorityOrderingMetrics,
    calculate_coefficient_of_variation,
    calculate_jain_fairness_index,
    calculate_max_min_fairness,
)
from futurnal.orchestrator.models import JobPriority, JobType


def test_jain_fairness_index_perfect_fairness():
    """Test JFI calculation with perfectly equal throughputs."""
    throughputs = [10.0, 10.0, 10.0, 10.0]
    jfi = calculate_jain_fairness_index(throughputs)
    assert jfi == 1.0


def test_jain_fairness_index_single_value():
    """Test JFI with single value (should be perfectly fair)."""
    throughputs = [5.0]
    jfi = calculate_jain_fairness_index(throughputs)
    assert jfi == 1.0


def test_jain_fairness_index_empty_list():
    """Test JFI with empty list (should return 1.0)."""
    throughputs = []
    jfi = calculate_jain_fairness_index(throughputs)
    assert jfi == 1.0


def test_jain_fairness_index_zero_throughputs():
    """Test JFI with all zero throughputs (should return 1.0)."""
    throughputs = [0.0, 0.0, 0.0]
    jfi = calculate_jain_fairness_index(throughputs)
    assert jfi == 1.0


def test_jain_fairness_index_unequal_distribution():
    """Test JFI with unequal throughputs (one starved connector)."""
    # One connector gets nothing, others get equal share
    throughputs = [10.0, 10.0, 10.0, 0.0]
    jfi = calculate_jain_fairness_index(throughputs)
    # With 4 entities and one getting 0, JFI should be less than 1.0
    assert 0.0 < jfi < 1.0
    # Specifically, (30)^2 / (4 * (100+100+100+0)) = 900/1200 = 0.75
    assert abs(jfi - 0.75) < 0.01


def test_jain_fairness_index_highly_unequal():
    """Test JFI with highly unequal distribution."""
    # One connector dominates
    throughputs = [100.0, 1.0, 1.0, 1.0]
    jfi = calculate_jain_fairness_index(throughputs)
    # Should show low fairness
    assert jfi < 0.5


def test_max_min_fairness_perfect():
    """Test max-min fairness with equal throughputs."""
    throughputs = [5.0, 5.0, 5.0]
    fairness = calculate_max_min_fairness(throughputs)
    assert fairness == 1.0


def test_max_min_fairness_unequal():
    """Test max-min fairness with unequal throughputs."""
    throughputs = [10.0, 5.0, 2.0]
    fairness = calculate_max_min_fairness(throughputs)
    # min/max = 2.0/10.0 = 0.2
    assert abs(fairness - 0.2) < 0.01


def test_max_min_fairness_empty():
    """Test max-min fairness with empty list."""
    throughputs = []
    fairness = calculate_max_min_fairness(throughputs)
    assert fairness == 1.0


def test_max_min_fairness_zero_max():
    """Test max-min fairness with zero max throughput."""
    throughputs = [0.0, 0.0]
    fairness = calculate_max_min_fairness(throughputs)
    assert fairness == 1.0


def test_coefficient_of_variation_identical_values():
    """Test CoV with identical values (perfect equality)."""
    values = [10.0, 10.0, 10.0, 10.0]
    cv = calculate_coefficient_of_variation(values)
    assert cv == 0.0


def test_coefficient_of_variation_variable_values():
    """Test CoV with variable values."""
    values = [10.0, 20.0, 30.0]
    cv = calculate_coefficient_of_variation(values)
    # Mean = 20, std = sqrt(((10-20)^2 + (20-20)^2 + (30-20)^2)/3) = sqrt(200/3) ≈ 8.165
    # CV = 8.165/20 ≈ 0.408
    assert 0.4 < cv < 0.45


def test_coefficient_of_variation_empty():
    """Test CoV with empty list."""
    values = []
    cv = calculate_coefficient_of_variation(values)
    assert cv == 0.0


def test_coefficient_of_variation_zero_mean():
    """Test CoV with zero mean."""
    values = [0.0, 0.0, 0.0]
    cv = calculate_coefficient_of_variation(values)
    assert cv == 0.0


def test_connector_load_creation():
    """Test ConnectorLoad dataclass creation."""
    load = ConnectorLoad(
        connector_type=JobType.LOCAL_FILES,
        jobs_per_minute=10,
        priority_distribution={JobPriority.NORMAL: 1.0},
        avg_job_size_bytes=1_000_000,
        avg_job_duration_seconds=5.0,
    )
    assert load.connector_type == JobType.LOCAL_FILES
    assert load.jobs_per_minute == 10
    assert load.avg_job_size_bytes == 1_000_000


def test_load_test_config_creation():
    """Test LoadTestConfig dataclass creation with defaults."""
    config = LoadTestConfig(
        name="test_scenario",
        duration_seconds=60,
        connectors=[],
    )
    assert config.name == "test_scenario"
    assert config.duration_seconds == 60
    assert config.target_throughput_mbps == 5.0  # default
    assert config.max_queue_depth == 1000  # default
    assert config.worker_count == 8  # default


def test_connector_metrics_creation():
    """Test ConnectorMetrics dataclass creation."""
    metrics = ConnectorMetrics(
        connector_type=JobType.OBSIDIAN_VAULT,
        jobs_completed=100,
        bytes_processed=10_000_000,
        total_duration_seconds=60.0,
        avg_job_latency_seconds=2.5,
        throughput_mbps=1.67,
        worker_time_seconds=250.0,
    )
    assert metrics.connector_type == JobType.OBSIDIAN_VAULT
    assert metrics.jobs_completed == 100
    assert metrics.throughput_mbps == 1.67


def test_fairness_metrics_is_fair_with_good_jfi():
    """Test FairnessMetrics.is_fair() with good JFI and no starvation."""
    metrics = FairnessMetrics(
        connector_metrics={},
        jain_fairness_index=0.85,
        max_min_fairness=0.7,
        coefficient_of_variation=0.2,
        starved_connectors=[],
    )
    assert metrics.is_fair(threshold=0.8) is True


def test_fairness_metrics_is_fair_with_low_jfi():
    """Test FairnessMetrics.is_fair() with low JFI."""
    metrics = FairnessMetrics(
        connector_metrics={},
        jain_fairness_index=0.6,
        max_min_fairness=0.5,
        coefficient_of_variation=0.5,
        starved_connectors=[],
    )
    assert metrics.is_fair(threshold=0.8) is False


def test_fairness_metrics_is_fair_with_starvation():
    """Test FairnessMetrics.is_fair() with starvation despite good JFI."""
    metrics = FairnessMetrics(
        connector_metrics={},
        jain_fairness_index=0.9,
        max_min_fairness=0.8,
        coefficient_of_variation=0.1,
        starved_connectors=[JobType.IMAP_MAILBOX],
    )
    # Even with good JFI, starvation makes it unfair
    assert metrics.is_fair(threshold=0.8) is False


def test_fairness_metrics_custom_threshold():
    """Test FairnessMetrics.is_fair() with custom threshold."""
    metrics = FairnessMetrics(
        connector_metrics={},
        jain_fairness_index=0.75,
        max_min_fairness=0.6,
        coefficient_of_variation=0.3,
        starved_connectors=[],
    )
    assert metrics.is_fair(threshold=0.7) is True
    assert metrics.is_fair(threshold=0.8) is False


def test_priority_ordering_metrics_valid_ordering():
    """Test PriorityOrderingMetrics with valid priority ordering."""
    metrics = PriorityOrderingMetrics(
        jobs_by_priority={
            JobPriority.HIGH: 10,
            JobPriority.NORMAL: 50,
            JobPriority.LOW: 100,
        },
        avg_latency_by_priority={
            JobPriority.HIGH: 1.0,
            JobPriority.NORMAL: 2.5,
            JobPriority.LOW: 5.0,
        },
        priority_inversions=2,
    )
    assert metrics.priority_ordering_valid() is True


def test_priority_ordering_metrics_invalid_ordering():
    """Test PriorityOrderingMetrics with invalid priority ordering."""
    metrics = PriorityOrderingMetrics(
        jobs_by_priority={
            JobPriority.HIGH: 10,
            JobPriority.NORMAL: 50,
            JobPriority.LOW: 100,
        },
        avg_latency_by_priority={
            JobPriority.HIGH: 5.0,  # HIGH slower than NORMAL - invalid!
            JobPriority.NORMAL: 2.5,
            JobPriority.LOW: 6.0,
        },
        priority_inversions=50,
    )
    assert metrics.priority_ordering_valid() is False


def test_priority_ordering_metrics_missing_priorities():
    """Test PriorityOrderingMetrics with missing priority levels."""
    metrics = PriorityOrderingMetrics(
        jobs_by_priority={
            JobPriority.NORMAL: 50,
        },
        avg_latency_by_priority={
            JobPriority.NORMAL: 2.5,
        },
        priority_inversions=0,
    )
    # Missing priorities default to inf, making the ordering check fail
    # This is correct behavior - can't validate ordering without all priorities
    assert metrics.priority_ordering_valid() is False
