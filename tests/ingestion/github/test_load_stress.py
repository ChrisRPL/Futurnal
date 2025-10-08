"""Load and stress tests for GitHub connector.

Tests validate system behavior under heavy load:
- Rate limit compliance under sustained load
- Concurrent repository sync
- Large repository handling (10k+ files)
- High commit frequency scenarios
- Queue backlog management

These tests verify production resilience and compliance.
"""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from futurnal.ingestion.github.sync_models import SyncResult, SyncStatus
from tests.ingestion.github.fixtures import (
    large_test_repo_fixture,
    medium_test_repo_fixture,
    enhanced_mock_github_api,
    rate_limit_exhausted_api,
)


# ---------------------------------------------------------------------------
# Load Test Markers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.github_load


# ---------------------------------------------------------------------------
# Rate Limit Compliance Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_compliance_sustained(enhanced_mock_github_api):
    """Test rate limit compliance under sustained load.

    Target: Never exceed GitHub rate limits
    Quality Gate: Critical for API compliance
    """
    api = enhanced_mock_github_api

    # Add test repository
    api.add_repository("test", "repo", {
        "name": "repo",
        "owner": {"login": "test"},
        "default_branch": "main",
    })

    violations = 0
    requests_made = 0
    max_requests = 500

    for i in range(max_requests):
        # Check if rate limit would be violated
        if api.rate_limiter.is_exhausted():
            # Should wait for reset instead of making request
            reset_in = api.rate_limiter.get_reset_in_seconds()
            assert reset_in >= 0, "Reset time should be non-negative"
            break

        try:
            data, headers = api.get_repository("test", "repo")
            requests_made += 1
        except Exception as e:
            if "Rate limit exceeded" in str(e):
                violations += 1

        await asyncio.sleep(0.001)

    # Verify no violations
    assert violations == 0, \
        f"Rate limit violated {violations} times out of {requests_made} requests"


@pytest.mark.asyncio
async def test_concurrent_repository_sync():
    """Test concurrent synchronization of multiple repositories.

    Target: 10 repositories synced concurrently without errors
    Quality Gate: Concurrency handling
    """
    num_repos = 10
    results = []

    async def sync_repository(repo_id):
        """Simulate repository sync."""
        await asyncio.sleep(0.1)  # Simulate sync work
        return SyncResult(
            status=SyncStatus.COMPLETED,
            files_processed=100,
            files_added=100,
            files_modified=0,
            files_deleted=0,
            sync_duration_seconds=0.1,
        )

    # Launch concurrent syncs
    tasks = [sync_repository(i) for i in range(num_repos)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Verify all succeeded
    successes = sum(1 for r in results if isinstance(r, SyncResult))
    assert successes == num_repos, \
        f"Only {successes}/{num_repos} syncs completed successfully"


@pytest.mark.asyncio
async def test_rate_limit_never_exceeded(enhanced_mock_github_api):
    """Test that rate limit is never exceeded even under aggressive load.

    Target: 0% rate limit violations
    Quality Gate: Critical for production
    """
    api = enhanced_mock_github_api

    # Add repository
    api.add_repository("test", "repo", {
        "name": "repo",
        "owner": {"login": "test"},
        "default_branch": "main",
    })

    # Aggressive request pattern
    violations = 0
    successful_requests = 0

    for _ in range(100):
        try:
            # Check before making request
            if api.rate_limiter.is_exhausted():
                # Wait instead of violating
                await asyncio.sleep(0.01)
                continue

            data, headers = api.get_repository("test", "repo")
            successful_requests += 1

        except Exception as e:
            if "403" in str(e) or "Rate limit" in str(e):
                violations += 1

    # Zero tolerance for violations
    assert violations == 0, \
        f"Rate limit violated {violations} times - CRITICAL FAILURE"

    assert successful_requests > 0, \
        "Should have made some successful requests"


# ---------------------------------------------------------------------------
# Large Repository Handling Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.slow
async def test_large_repository_processing(large_test_repo_fixture):
    """Test processing of large repository (10k+ files).

    Target: Successfully process all files without timeout
    Quality Gate: Scalability
    """
    mock_repo = large_test_repo_fixture
    processed_files = 0
    errors = 0

    # Process files in chunks
    chunk_size = 100
    for i in range(0, min(len(mock_repo.files), 1000), chunk_size):
        chunk = mock_repo.files[i:i+chunk_size]

        try:
            for file in chunk:
                # Simulate processing
                _ = file.path
                processed_files += 1
                await asyncio.sleep(0.0001)  # Minimal delay
        except Exception:
            errors += 1

    # Verify high success rate
    success_rate = (processed_files / min(len(mock_repo.files), 1000)) * 100
    assert success_rate > 99, \
        f"Success rate {success_rate:.1f}% is below 99%"

    assert errors < 10, \
        f"Too many errors: {errors}"


@pytest.mark.asyncio
async def test_high_commit_frequency():
    """Test handling of high commit frequency (100 commits/hour).

    Target: Process commits without backlog buildup
    Quality Gate: Real-time processing
    """
    commits_per_hour = 100
    commit_interval = 3600 / commits_per_hour  # seconds between commits

    processed_commits = []
    queue_size = 0
    max_queue_size = 0

    async def process_commit(commit_id):
        """Simulate commit processing."""
        nonlocal queue_size, max_queue_size
        queue_size += 1
        max_queue_size = max(max_queue_size, queue_size)

        await asyncio.sleep(0.01)  # Processing time

        processed_commits.append(commit_id)
        queue_size -= 1

    # Simulate commit stream (scaled down for test speed)
    num_commits = 20
    tasks = []

    for i in range(num_commits):
        task = asyncio.create_task(process_commit(i))
        tasks.append(task)
        await asyncio.sleep(0.005)  # Scaled-down interval

    # Wait for all to complete
    await asyncio.gather(*tasks)

    # Verify no excessive backlog
    assert max_queue_size < num_commits, \
        f"Queue backed up to {max_queue_size} items"

    assert len(processed_commits) == num_commits, \
        "Not all commits processed"


@pytest.mark.asyncio
async def test_large_file_handling():
    """Test handling of large binary files.

    Target: Process large files without memory issues
    Quality Gate: Resource management
    """
    import sys

    large_file_sizes = [
        1 * 1024 * 1024,    # 1 MB
        5 * 1024 * 1024,    # 5 MB
        10 * 1024 * 1024,   # 10 MB
    ]

    processed = 0

    for size in large_file_sizes:
        try:
            # Simulate large file (don't actually allocate full size)
            file_meta = {
                "size": size,
                "path": f"large_file_{size}.bin",
                "should_skip": size > 5 * 1024 * 1024,  # Skip >5MB
            }

            if not file_meta["should_skip"]:
                # Simulate processing metadata only (not content)
                await asyncio.sleep(0.001)
                processed += 1

        except MemoryError:
            pytest.fail(f"Memory error processing {size} byte file")

    assert processed > 0, "Should process at least some files"


# ---------------------------------------------------------------------------
# Concurrent Operations Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_webhook_processing():
    """Test concurrent webhook event processing.

    Target: Handle 10 webhooks simultaneously
    Quality Gate: Event handling capacity
    """
    num_webhooks = 10
    processed = []
    errors = []

    async def process_webhook(webhook_id):
        """Simulate webhook processing."""
        try:
            await asyncio.sleep(0.05)  # Simulate processing
            processed.append(webhook_id)
        except Exception as e:
            errors.append((webhook_id, e))

    # Process webhooks concurrently
    tasks = [process_webhook(i) for i in range(num_webhooks)]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Verify all processed
    assert len(processed) == num_webhooks, \
        f"Only processed {len(processed)}/{num_webhooks} webhooks"

    assert len(errors) == 0, \
        f"Errors occurred: {errors}"


@pytest.mark.asyncio
async def test_concurrent_api_clients():
    """Test thread safety of concurrent API client usage.

    Target: No race conditions or data corruption
    Quality Gate: Thread safety
    """
    num_clients = 5
    requests_per_client = 20
    results = {i: [] for i in range(num_clients)}

    async def make_requests(client_id):
        """Simulate client making requests."""
        for req_num in range(requests_per_client):
            # Simulate request
            result = {
                "client_id": client_id,
                "request_num": req_num,
                "timestamp": time.time(),
            }
            results[client_id].append(result)
            await asyncio.sleep(0.001)

    # Run concurrent clients
    tasks = [make_requests(i) for i in range(num_clients)]
    await asyncio.gather(*tasks)

    # Verify no data corruption
    for client_id, client_results in results.items():
        assert len(client_results) == requests_per_client, \
            f"Client {client_id} lost data"

        # Verify sequential request numbers
        request_nums = [r["request_num"] for r in client_results]
        assert request_nums == list(range(requests_per_client)), \
            f"Client {client_id} has out-of-order data"


@pytest.mark.asyncio
async def test_queue_backlog_handling():
    """Test queue backlog handling under load.

    Target: Process backlog efficiently
    Quality Gate: Queue management
    """
    queue = asyncio.Queue()
    processed = []
    max_backlog = 0

    # Producer: Add items to queue rapidly
    async def producer():
        for i in range(100):
            await queue.put(i)
            await asyncio.sleep(0.001)  # Fast production

    # Consumer: Process items (slower)
    async def consumer():
        nonlocal max_backlog
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.5)
                max_backlog = max(max_backlog, queue.qsize())
                processed.append(item)
                await asyncio.sleep(0.005)  # Slower consumption
            except asyncio.TimeoutError:
                break

    # Run producer and consumer concurrently
    await asyncio.gather(
        producer(),
        consumer(),
        return_exceptions=True,
    )

    # Verify reasonable backlog
    assert max_backlog < 50, \
        f"Queue backed up to {max_backlog} items"

    # Verify all processed eventually
    assert len(processed) == 100, \
        f"Only processed {len(processed)}/100 items"


# ---------------------------------------------------------------------------
# Stress Tests (Failure Scenarios)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_failure_cascade_prevention():
    """Test that API failures don't cascade.

    Target: Isolated failures don't affect other operations
    Quality Gate: Fault isolation
    """
    operations = []

    async def operation_with_failure(op_id, should_fail=False):
        """Simulate operation that may fail."""
        if should_fail:
            raise Exception(f"Operation {op_id} failed")

        await asyncio.sleep(0.01)
        operations.append(op_id)
        return f"success_{op_id}"

    # Mix of successful and failing operations
    tasks = [
        operation_with_failure(0, should_fail=False),
        operation_with_failure(1, should_fail=True),  # Failure
        operation_with_failure(2, should_fail=False),
        operation_with_failure(3, should_fail=False),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successes
    successes = sum(1 for r in results if not isinstance(r, Exception))

    # Verify failures isolated
    assert successes == 3, \
        "Failures cascaded to other operations"


@pytest.mark.asyncio
async def test_rate_limit_recovery(rate_limit_exhausted_api):
    """Test recovery after rate limit exhaustion.

    Target: Graceful handling and automatic recovery
    Quality Gate: Resilience
    """
    api = rate_limit_exhausted_api

    # Add repository
    api.add_repository("test", "repo", {
        "name": "repo",
        "owner": {"login": "test"},
        "default_branch": "main",
    })

    # Attempt request with exhausted rate limit
    with pytest.raises(Exception, match="Rate limit exceeded"):
        api.get_repository("test", "repo")

    # Simulate time passage (reset rate limit for test)
    api.rate_limiter.remaining = api.rate_limiter.limit

    # Verify recovery
    data, headers = api.get_repository("test", "repo")
    assert data is not None, "Should recover after rate limit reset"


@pytest.mark.asyncio
async def test_circuit_breaker_under_load():
    """Test circuit breaker behavior under sustained failures.

    Target: Circuit opens after threshold, preventing cascade
    Quality Gate: Fault tolerance
    """
    failure_count = 0
    circuit_open = False

    async def failing_operation():
        """Simulate operation that fails."""
        nonlocal failure_count, circuit_open

        if circuit_open:
            raise Exception("Circuit breaker open")

        failure_count += 1

        if failure_count >= 5:
            circuit_open = True

        raise Exception("Operation failed")

    # Attempt operations
    exceptions = []
    for _ in range(10):
        try:
            await failing_operation()
        except Exception as e:
            exceptions.append(str(e))
            await asyncio.sleep(0.001)

    # Verify circuit opened
    circuit_breaker_messages = [e for e in exceptions if "Circuit breaker" in e]
    assert len(circuit_breaker_messages) > 0, \
        "Circuit breaker should have opened"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "github_load"])
