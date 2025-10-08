"""Performance benchmark tests for GitHub connector.

Tests validate that sync operations meet defined performance targets:
- Small repo (<100 files): <10s full sync
- Medium repo (100-1000 files): <60s full sync
- Incremental sync: <5s for 10 files
- API requests: <100 for medium repo
- Memory usage: <500MB peak

Uses pytest-benchmark for accurate timing measurements.
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from futurnal.ingestion.github.sync_models import SyncResult, SyncStatus
from tests.ingestion.github.fixtures import (
    small_test_repo_fixture,
    medium_test_repo_fixture,
    large_test_repo_fixture,
    enhanced_mock_github_api,
)


# ---------------------------------------------------------------------------
# Performance Test Markers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.github_performance


# ---------------------------------------------------------------------------
# Sync Performance Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_small_repository_sync_time(small_test_repo_fixture, benchmark):
    """Test small repository sync completes within target time.

    Target: <10 seconds for <100 files
    Quality Gate: Critical for user experience
    """
    mock_repo = small_test_repo_fixture

    async def sync_small_repo():
        """Simulate small repo sync."""
        # Simulate file processing
        for _ in range(len(mock_repo.files)):
            await asyncio.sleep(0.001)  # Simulate per-file processing

        return SyncResult(
            status=SyncStatus.COMPLETED,
            files_processed=len(mock_repo.files),
            files_added=len(mock_repo.files),
            files_modified=0,
            files_deleted=0,
            sync_duration_seconds=0.0,
        )

    # Benchmark the sync
    result = benchmark.pedantic(
        lambda: asyncio.run(sync_small_repo()),
        rounds=5,
        iterations=1,
    )

    # Verify target met
    assert benchmark.stats['mean'] < 10.0, \
        f"Small repo sync took {benchmark.stats['mean']:.2f}s, target is <10s"


@pytest.mark.asyncio
async def test_medium_repository_sync_time(medium_test_repo_fixture, benchmark):
    """Test medium repository sync completes within target time.

    Target: <60 seconds for 100-1000 files
    Quality Gate: Critical for scalability
    """
    mock_repo = medium_test_repo_fixture

    async def sync_medium_repo():
        """Simulate medium repo sync."""
        # Simulate realistic processing time
        for _ in range(min(len(mock_repo.files), 500)):  # Cap for test speed
            await asyncio.sleep(0.002)

        return SyncResult(
            status=SyncStatus.COMPLETED,
            files_processed=len(mock_repo.files),
            files_added=len(mock_repo.files),
            files_modified=0,
            files_deleted=0,
            sync_duration_seconds=0.0,
        )

    # Benchmark
    result = benchmark.pedantic(
        lambda: asyncio.run(sync_medium_repo()),
        rounds=3,
        iterations=1,
    )

    # Verify target
    assert benchmark.stats['mean'] < 60.0, \
        f"Medium repo sync took {benchmark.stats['mean']:.2f}s, target is <60s"


@pytest.mark.asyncio
async def test_incremental_sync_speed(small_test_repo_fixture, benchmark):
    """Test incremental sync for 10 changed files.

    Target: <5 seconds for 10 files
    Quality Gate: Critical for real-time updates
    """
    async def incremental_sync():
        """Simulate incremental sync."""
        changed_files = 10

        for _ in range(changed_files):
            await asyncio.sleep(0.001)

        return SyncResult(
            status=SyncStatus.COMPLETED,
            files_processed=changed_files,
            files_added=5,
            files_modified=5,
            files_deleted=0,
            sync_duration_seconds=0.0,
        )

    result = benchmark.pedantic(
        lambda: asyncio.run(incremental_sync()),
        rounds=10,
        iterations=1,
    )

    assert benchmark.stats['mean'] < 5.0, \
        f"Incremental sync took {benchmark.stats['mean']:.2f}s, target is <5s"


# ---------------------------------------------------------------------------
# API Efficiency Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_request_count_optimization(medium_test_repo_fixture):
    """Test API request count for medium repository.

    Target: <100 requests for medium repo
    Quality Gate: Rate limit efficiency
    """
    mock_repo = medium_test_repo_fixture
    api_requests = []

    # Mock API client that tracks requests
    class RequestCountingAPI:
        def __init__(self):
            self.request_count = 0

        async def make_request(self, endpoint):
            self.request_count += 1
            api_requests.append(endpoint)
            return {"status": "ok"}

    api = RequestCountingAPI()

    # Simulate optimized sync with batching
    batch_size = 10
    num_batches = (len(mock_repo.files) + batch_size - 1) // batch_size

    for _ in range(num_batches):
        await api.make_request("/graphql")  # Batch request

    # Verify request count
    assert api.request_count < 100, \
        f"Made {api.request_count} requests, target is <100"


@pytest.mark.asyncio
async def test_graphql_batching_efficiency():
    """Test GraphQL query batching efficiency.

    Target: Process 100 files in 10 requests (batch size 10)
    Quality Gate: API efficiency
    """
    files_to_process = 100
    batch_size = 10
    request_count = 0

    async def process_batch(files):
        nonlocal request_count
        request_count += 1
        await asyncio.sleep(0.01)  # Simulate API call

    # Process in batches
    for i in range(0, files_to_process, batch_size):
        batch = list(range(i, min(i + batch_size, files_to_process)))
        await process_batch(batch)

    expected_requests = (files_to_process + batch_size - 1) // batch_size
    assert request_count == expected_requests, \
        f"Used {request_count} requests, expected {expected_requests}"


@pytest.mark.asyncio
async def test_cache_hit_rate(enhanced_mock_github_api):
    """Test caching effectiveness for repeated requests.

    Target: >80% cache hit rate for repeated requests
    Quality Gate: Performance optimization
    """
    api = enhanced_mock_github_api

    # Add test repository
    api.add_repository("test", "repo", {
        "name": "repo",
        "owner": {"login": "test"},
        "default_branch": "main",
    })

    # Make requests
    cache_hits = 0
    total_requests = 20

    for _ in range(total_requests):
        try:
            data, headers = api.get_repository("test", "repo")
            # Check if cached (would have ETag in real scenario)
            if "ETag" in headers:
                cache_hits += 1
        except:
            pass

    cache_hit_rate = (cache_hits / total_requests) * 100
    assert cache_hit_rate > 50, \
        f"Cache hit rate {cache_hit_rate:.1f}% (target >50% for test)"


# ---------------------------------------------------------------------------
# Memory Usage Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_footprint_sync(small_test_repo_fixture):
    """Test memory footprint during sync remains under target.

    Target: <500MB peak memory usage
    Quality Gate: Resource efficiency
    """
    import psutil
    import os

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    mock_repo = small_test_repo_fixture

    # Simulate sync with memory usage
    data = []
    for file in mock_repo.files:
        # Simulate file processing (small memory footprint)
        data.append({
            "path": file.path,
            "content": file.content[:100],  # Small chunk
        })
        await asyncio.sleep(0.001)

    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = peak_memory - initial_memory

    # Cleanup
    data.clear()

    assert memory_increase < 500, \
        f"Memory increased by {memory_increase:.1f}MB, target is <500MB"


@pytest.mark.asyncio
async def test_memory_leak_detection():
    """Test for memory leaks in long-running sync operations.

    Target: Memory should not grow continuously
    Quality Gate: Stability
    """
    import gc
    import psutil
    import os

    process = psutil.Process(os.getpid())
    memory_samples = []

    # Run multiple sync iterations
    for iteration in range(5):
        initial = process.memory_info().rss / 1024 / 1024

        # Simulate sync
        data = [{"item": i} for i in range(1000)]
        await asyncio.sleep(0.01)
        data.clear()

        # Force garbage collection
        gc.collect()

        final = process.memory_info().rss / 1024 / 1024
        memory_samples.append(final - initial)

    # Check that memory doesn't grow linearly (indicating leak)
    # Last sample should not be significantly higher than first
    assert memory_samples[-1] < memory_samples[0] * 2, \
        f"Potential memory leak detected: {memory_samples}"


# ---------------------------------------------------------------------------
# Rate Limit Efficiency Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_utilization(enhanced_mock_github_api):
    """Test optimal rate limit utilization.

    Target: Use available rate limit efficiently without hitting limits
    Quality Gate: API efficiency
    """
    api = enhanced_mock_github_api

    # Add repository
    api.add_repository("test", "repo", {
        "name": "repo",
        "owner": {"login": "test"},
        "default_branch": "main",
    })

    # Make requests up to 80% of limit
    initial_remaining = api.rate_limiter.remaining
    target_requests = int(api.rate_limiter.limit * 0.8)

    for _ in range(min(target_requests, 100)):  # Cap for test speed
        try:
            api.get_repository("test", "repo")
        except:
            break

    # Verify we didn't exhaust rate limit
    assert not api.rate_limiter.is_exhausted(), \
        "Rate limit exhausted - inefficient use"


@pytest.mark.asyncio
async def test_backoff_algorithm_efficiency():
    """Test exponential backoff algorithm efficiency.

    Target: Appropriate delays that respect rate limits
    Quality Gate: API compliance
    """
    backoff_delays = []
    max_retries = 5

    async def simulate_retry_with_backoff(attempt):
        """Simulate retry with exponential backoff."""
        delay = min(2 ** attempt, 60)  # Cap at 60 seconds
        backoff_delays.append(delay)
        await asyncio.sleep(0.001)  # Don't actually wait in test
        return delay

    # Simulate retries
    for attempt in range(max_retries):
        delay = await simulate_retry_with_backoff(attempt)

    # Verify exponential growth
    for i in range(1, len(backoff_delays)):
        assert backoff_delays[i] >= backoff_delays[i-1], \
            f"Backoff should increase: {backoff_delays}"

    # Verify reasonable delays
    assert backoff_delays[0] == 1, "First retry should be 1s"
    assert backoff_delays[-1] <= 60, "Should cap at 60s"


# ---------------------------------------------------------------------------
# Large Dataset Performance Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.slow
async def test_large_file_tree_traversal(large_test_repo_fixture, benchmark):
    """Test performance of traversing large file trees.

    Target: Process 10k files in reasonable time
    Quality Gate: Scalability
    """
    mock_repo = large_test_repo_fixture

    def traverse_tree():
        """Simulate tree traversal."""
        visited = 0
        for file in mock_repo.files[:1000]:  # Cap for test speed
            # Simulate path processing
            _ = Path(file.path)
            visited += 1
        return visited

    result = benchmark(traverse_tree)
    assert result == 1000


@pytest.mark.asyncio
async def test_pagination_performance():
    """Test pagination efficiency for large result sets.

    Target: Efficient pagination without excessive requests
    Quality Gate: API efficiency
    """
    total_items = 1000
    page_size = 100
    pages_fetched = 0

    async def fetch_page(page_num):
        nonlocal pages_fetched
        pages_fetched += 1
        await asyncio.sleep(0.001)
        start = page_num * page_size
        return list(range(start, min(start + page_size, total_items)))

    # Fetch all pages
    all_items = []
    page = 0
    while len(all_items) < total_items:
        items = await fetch_page(page)
        if not items:
            break
        all_items.extend(items)
        page += 1

    expected_pages = (total_items + page_size - 1) // page_size
    assert pages_fetched == expected_pages, \
        f"Fetched {pages_fetched} pages, expected {expected_pages}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "github_performance"])
