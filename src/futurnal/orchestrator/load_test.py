"""Load testing data models and fairness metrics for orchestrator validation.

This module provides the infrastructure for validating multi-connector concurrent
execution, priority ordering, resource fairness, and throughput under load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from .models import JobPriority, JobType


@dataclass
class ConnectorLoad:
    """Load specification for a connector.

    Attributes:
        connector_type: Type of connector being tested
        jobs_per_minute: Rate of job generation
        priority_distribution: Distribution of priorities (e.g., {HIGH: 0.2, NORMAL: 0.7, LOW: 0.1})
        avg_job_size_bytes: Average job size in bytes
        avg_job_duration_seconds: Expected average job duration
    """
    connector_type: JobType
    jobs_per_minute: int
    priority_distribution: Dict[JobPriority, float]
    avg_job_size_bytes: int
    avg_job_duration_seconds: float


@dataclass
class LoadTestConfig:
    """Configuration for load test scenario.

    Attributes:
        name: Test scenario name
        duration_seconds: Test duration in seconds
        connectors: List of connector load specifications
        target_throughput_mbps: Target throughput in MB/s (default: 5.0)
        max_queue_depth: Maximum queue depth (default: 1000)
        worker_count: Number of concurrent workers (default: 8)
    """
    name: str
    duration_seconds: int
    connectors: List[ConnectorLoad]
    target_throughput_mbps: float = 5.0
    max_queue_depth: int = 1000
    worker_count: int = 8


@dataclass
class ConnectorMetrics:
    """Per-connector execution metrics.

    Attributes:
        connector_type: Type of connector
        jobs_completed: Number of jobs successfully completed
        bytes_processed: Total bytes processed
        total_duration_seconds: Total test duration
        avg_job_latency_seconds: Average job latency (queue time + execution)
        throughput_mbps: Throughput in MB/s
        worker_time_seconds: Total worker time allocated to this connector
    """
    connector_type: JobType
    jobs_completed: int
    bytes_processed: int
    total_duration_seconds: float
    avg_job_latency_seconds: float
    throughput_mbps: float
    worker_time_seconds: float


@dataclass
class FairnessMetrics:
    """Metrics for evaluating scheduling fairness.

    Uses multiple fairness measures:
    - Jain's Fairness Index: Overall fairness score [1/n, 1.0]
    - Max-Min Fairness: Ratio of min to max throughput [0.0, 1.0]
    - Coefficient of Variation: Standard deviation / mean

    Attributes:
        connector_metrics: Per-connector execution metrics
        jain_fairness_index: Overall fairness (1.0 = perfectly fair)
        max_min_fairness: Min/max throughput ratio (1.0 = perfectly fair)
        coefficient_of_variation: Throughput variability measure
        starved_connectors: List of connectors with <10% expected throughput
    """
    connector_metrics: Dict[JobType, ConnectorMetrics]
    jain_fairness_index: float
    max_min_fairness: float
    coefficient_of_variation: float
    starved_connectors: List[JobType] = field(default_factory=list)

    def is_fair(self, threshold: float = 0.8) -> bool:
        """Check if scheduling is fair.

        Args:
            threshold: Minimum JFI threshold for fairness (default: 0.8)

        Returns:
            True if JFI >= threshold and no connectors are starved
        """
        return (
            self.jain_fairness_index >= threshold and
            len(self.starved_connectors) == 0
        )


@dataclass
class PriorityOrderingMetrics:
    """Metrics for priority ordering validation.

    Validates that higher priority jobs have lower latency.

    Attributes:
        jobs_by_priority: Count of jobs per priority level
        avg_latency_by_priority: Average latency per priority level
        priority_inversions: Count of LOW jobs completed before HIGH jobs
    """
    jobs_by_priority: Dict[JobPriority, int]
    avg_latency_by_priority: Dict[JobPriority, float]
    priority_inversions: int

    def priority_ordering_valid(self) -> bool:
        """Check if higher priority jobs have lower latency.

        Returns:
            True if HIGH latency <= NORMAL latency <= LOW latency
        """
        high_latency = self.avg_latency_by_priority.get(JobPriority.HIGH, float("inf"))
        normal_latency = self.avg_latency_by_priority.get(JobPriority.NORMAL, float("inf"))
        low_latency = self.avg_latency_by_priority.get(JobPriority.LOW, float("inf"))

        return high_latency <= normal_latency <= low_latency


def calculate_jain_fairness_index(throughputs: List[float]) -> float:
    """Calculate Jain's Fairness Index.

    JFI = (sum(x_i))^2 / (n * sum(x_i^2))

    Range: [1/n, 1.0] where 1.0 is perfectly fair.
    A higher value indicates more fair resource allocation.

    Args:
        throughputs: List of throughput values for each entity

    Returns:
        Jain's Fairness Index in range [1/n, 1.0]
        Returns 1.0 for empty list or zero throughputs
    """
    if not throughputs:
        return 1.0

    n = len(throughputs)
    sum_x = sum(throughputs)
    sum_x_squared = sum(x ** 2 for x in throughputs)

    if sum_x_squared == 0:
        return 1.0

    return (sum_x ** 2) / (n * sum_x_squared)


def calculate_max_min_fairness(throughputs: List[float]) -> float:
    """Calculate max/min fairness ratio.

    Ratio of minimum to maximum throughput.

    Range: [0.0, 1.0] where 1.0 is perfectly fair.
    Measures the degree of inequality between best and worst cases.

    Args:
        throughputs: List of throughput values for each entity

    Returns:
        Max-min fairness ratio in range [0.0, 1.0]
        Returns 1.0 for empty list or zero max throughput
    """
    if not throughputs:
        return 1.0

    min_throughput = min(throughputs)
    max_throughput = max(throughputs)

    if max_throughput == 0:
        return 1.0

    return min_throughput / max_throughput


def calculate_coefficient_of_variation(values: List[float]) -> float:
    """Calculate coefficient of variation (CV).

    CV = standard_deviation / mean

    Measures relative variability. Lower values indicate more consistent allocation.
    CV = 0 indicates perfect equality.

    Args:
        values: List of numeric values

    Returns:
        Coefficient of variation (0 = perfect equality)
        Returns 0.0 for empty list or zero mean
    """
    if not values:
        return 0.0

    n = len(values)
    mean = sum(values) / n

    if mean == 0:
        return 0.0

    variance = sum((x - mean) ** 2 for x in values) / n
    std_dev = math.sqrt(variance)

    return std_dev / mean
