"""Load test framework runner for orchestrator validation.

This module provides the LoadTestRunner class for executing load tests
and collecting comprehensive metrics on orchestrator performance, fairness,
and priority ordering.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .load_test import (
    ConnectorLoad,
    ConnectorMetrics,
    FairnessMetrics,
    LoadTestConfig,
    calculate_coefficient_of_variation,
    calculate_jain_fairness_index,
    calculate_max_min_fairness,
)
from .models import IngestionJob, JobPriority, JobType
from .queue import JobQueue, JobStatus
from .scheduler import IngestionOrchestrator


class LoadTestRunner:
    """Framework for running load tests on the ingestion orchestrator.

    This runner generates jobs at configurable rates, monitors execution,
    and collects comprehensive metrics on throughput, fairness, and latency.
    """

    def __init__(
        self,
        orchestrator: IngestionOrchestrator,
        job_queue: JobQueue,
        source_name_map: Optional[Dict[JobType, str]] = None,
    ):
        """Initialize load test runner.

        Args:
            orchestrator: The orchestrator to test
            job_queue: Job queue for monitoring and metrics
            source_name_map: Optional mapping from JobType to registered source name
                           (e.g., {JobType.LOCAL_FILES: "local_test"})
        """
        self._orchestrator = orchestrator
        self._queue = job_queue
        self._start_time: Optional[float] = None
        self._generated_job_ids: List[str] = []
        self._source_name_map = source_name_map or {}

    async def run_test(self, config: LoadTestConfig) -> FairnessMetrics:
        """Execute load test and collect metrics.

        Args:
            config: Load test configuration

        Returns:
            FairnessMetrics containing test results
        """
        self._start_time = time.perf_counter()
        self._generated_job_ids = []

        # Start job generator tasks for each connector
        generator_tasks = [
            asyncio.create_task(
                self._generate_jobs(connector_load, config.duration_seconds)
            )
            for connector_load in config.connectors
        ]

        # Start orchestrator if not already running
        if not self._orchestrator._running:
            self._orchestrator.start()
            orchestrator_started = True
        else:
            orchestrator_started = False

        try:
            # Wait for test duration
            await asyncio.sleep(config.duration_seconds)

            # Stop job generators
            for task in generator_tasks:
                task.cancel()

            # Wait for all generator tasks to cancel
            await asyncio.gather(*generator_tasks, return_exceptions=True)

            # Wait for queue to drain (with timeout)
            await self._wait_for_completion(timeout=config.duration_seconds * 2)

            # Collect metrics
            return self._collect_metrics(config)

        except Exception:
            # Ensure cleanup on error
            if orchestrator_started and self._orchestrator._running:
                try:
                    await self._orchestrator.shutdown()
                except Exception:
                    pass
            raise
        finally:
            # Stop orchestrator if we started it
            if orchestrator_started and self._orchestrator._running:
                try:
                    await self._orchestrator.shutdown()
                except Exception:
                    # Swallow shutdown errors - we're done with the test
                    pass

    async def _generate_jobs(
        self,
        connector_load: ConnectorLoad,
        duration_seconds: int,
    ) -> None:
        """Generate jobs at specified rate.

        Args:
            connector_load: Connector load specification
            duration_seconds: Duration to generate jobs
        """
        interval = 60.0 / connector_load.jobs_per_minute
        end_time = time.perf_counter() + duration_seconds

        try:
            while time.perf_counter() < end_time:
                # Sample priority from distribution
                priority = self._sample_priority(connector_load.priority_distribution)

                job_id = str(uuid.uuid4())

                # Use mapped source name if available
                source_name = self._source_name_map.get(
                    connector_load.connector_type,
                    f"load_test_{connector_load.connector_type.value}"
                )

                job = IngestionJob(
                    job_id=job_id,
                    job_type=connector_load.connector_type,
                    payload={
                        "source_name": source_name,
                        "size_bytes": connector_load.avg_job_size_bytes,
                        "expected_duration": connector_load.avg_job_duration_seconds,
                        "load_test": True,
                    },
                    priority=priority,
                    scheduled_for=datetime.utcnow(),
                )
                self._queue.enqueue(job)
                self._generated_job_ids.append(job_id)

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            # Generator cancelled - normal shutdown
            pass

    async def _wait_for_completion(self, timeout: int) -> None:
        """Wait for queue to drain.

        Args:
            timeout: Maximum time to wait in seconds
        """
        end_time = time.perf_counter() + timeout

        while time.perf_counter() < end_time:
            pending = self._queue.pending_count()
            running = self._queue.running_count()

            if pending == 0 and running == 0:
                break

            await asyncio.sleep(0.5)

    def _sample_priority(self, distribution: Dict[JobPriority, float]) -> JobPriority:
        """Sample a priority from the distribution.

        Args:
            distribution: Priority distribution (probabilities must sum to 1.0)

        Returns:
            Sampled JobPriority
        """
        priorities = list(distribution.keys())
        weights = [distribution[p] for p in priorities]

        return random.choices(priorities, weights=weights, k=1)[0]

    def _collect_metrics(self, config: LoadTestConfig) -> FairnessMetrics:
        """Collect metrics from completed jobs.

        Args:
            config: Load test configuration

        Returns:
            FairnessMetrics with comprehensive test results
        """
        all_jobs = self._queue.snapshot()
        completed_jobs = [j for j in all_jobs if j["status"] == JobStatus.SUCCEEDED.value]

        # Filter to only jobs generated in this test
        test_jobs = [j for j in completed_jobs if j["job_id"] in self._generated_job_ids]

        # Group by connector
        connector_metrics = {}
        for connector_load in config.connectors:
            connector_jobs = [
                j for j in test_jobs
                if j["job_type"] == connector_load.connector_type.value
            ]

            if not connector_jobs:
                # No jobs completed for this connector
                metrics = ConnectorMetrics(
                    connector_type=connector_load.connector_type,
                    jobs_completed=0,
                    bytes_processed=0,
                    total_duration_seconds=config.duration_seconds,
                    avg_job_latency_seconds=0.0,
                    throughput_mbps=0.0,
                    worker_time_seconds=0.0,
                )
            else:
                bytes_processed = sum(
                    j["payload"].get("bytes_processed", 0) for j in connector_jobs
                )
                avg_latency = self._calculate_avg_latency(connector_jobs)
                throughput = self._calculate_throughput(
                    connector_jobs, config.duration_seconds
                )
                worker_time = sum(
                    self._calculate_job_duration(j) for j in connector_jobs
                )

                metrics = ConnectorMetrics(
                    connector_type=connector_load.connector_type,
                    jobs_completed=len(connector_jobs),
                    bytes_processed=bytes_processed,
                    total_duration_seconds=config.duration_seconds,
                    avg_job_latency_seconds=avg_latency,
                    throughput_mbps=throughput,
                    worker_time_seconds=worker_time,
                )

            connector_metrics[connector_load.connector_type] = metrics

        # Calculate fairness scores
        throughputs = [m.throughput_mbps for m in connector_metrics.values()]
        jfi = calculate_jain_fairness_index(throughputs)
        max_min = calculate_max_min_fairness(throughputs)
        cv = calculate_coefficient_of_variation(throughputs)

        # Detect starvation (connectors with <10% of expected throughput)
        starved_connectors = self._detect_starvation(
            connector_metrics, config.connectors
        )

        return FairnessMetrics(
            connector_metrics=connector_metrics,
            jain_fairness_index=jfi,
            max_min_fairness=max_min,
            coefficient_of_variation=cv,
            starved_connectors=starved_connectors,
        )

    def _calculate_avg_latency(self, jobs: List[dict]) -> float:
        """Calculate average job latency.

        Latency = time from creation to completion.

        Args:
            jobs: List of job dictionaries

        Returns:
            Average latency in seconds
        """
        if not jobs:
            return 0.0

        latencies = []
        for job in jobs:
            created_at = datetime.fromisoformat(job["created_at"])
            updated_at = datetime.fromisoformat(job["updated_at"])
            latency = (updated_at - created_at).total_seconds()
            latencies.append(latency)

        return sum(latencies) / len(latencies)

    def _calculate_throughput(self, jobs: List[dict], duration: float) -> float:
        """Calculate throughput in MB/s.

        Args:
            jobs: List of job dictionaries
            duration: Test duration in seconds

        Returns:
            Throughput in MB/s
        """
        if not jobs or duration == 0:
            return 0.0

        total_bytes = sum(j["payload"].get("bytes_processed", 0) for j in jobs)
        throughput_bps = total_bytes / duration
        throughput_mbps = throughput_bps / (1024 * 1024)

        return throughput_mbps

    def _calculate_job_duration(self, job: dict) -> float:
        """Calculate job execution duration.

        Args:
            job: Job dictionary

        Returns:
            Execution duration in seconds
        """
        # For load tests, we track the actual execution time
        # This is an approximation based on created_at and updated_at
        created_at = datetime.fromisoformat(job["created_at"])
        updated_at = datetime.fromisoformat(job["updated_at"])
        return (updated_at - created_at).total_seconds()

    def _detect_starvation(
        self,
        connector_metrics: Dict[JobType, ConnectorMetrics],
        connector_loads: List[ConnectorLoad],
    ) -> List[JobType]:
        """Detect starved connectors (<10% of expected throughput).

        Args:
            connector_metrics: Collected connector metrics
            connector_loads: Expected connector loads

        Returns:
            List of starved connector types
        """
        starved = []

        # Build expected throughput map
        expected_throughput = {}
        for load in connector_loads:
            # Expected throughput = jobs_per_minute * avg_job_size_bytes / 60
            expected_bps = (load.jobs_per_minute * load.avg_job_size_bytes) / 60.0
            expected_mbps = expected_bps / (1024 * 1024)
            expected_throughput[load.connector_type] = expected_mbps

        # Check each connector
        for connector_type, metrics in connector_metrics.items():
            expected = expected_throughput.get(connector_type, 0.0)
            if expected > 0:
                ratio = metrics.throughput_mbps / expected
                if ratio < 0.1:  # Less than 10% of expected
                    starved.append(connector_type)

        return starved

    def _calculate_cv(self, values: List[float]) -> float:
        """Calculate coefficient of variation.

        Args:
            values: List of numeric values

        Returns:
            Coefficient of variation
        """
        return calculate_coefficient_of_variation(values)
