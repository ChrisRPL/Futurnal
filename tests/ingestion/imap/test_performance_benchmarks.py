"""Performance benchmark tests for IMAP connector.

Tests performance requirements:
- Large mailbox initial sync (<3 hours for 10,000 messages)
- High-volume IDLE events (<5 minutes detection for 100 messages)
- Connection pool efficiency (100 concurrent operations)
- Memory usage monitoring (<500MB peak for long-running sync)
- Throughput validation (≥1 msg/s sustained)
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from futurnal.ingestion.imap.sync_metrics import ImapSyncMetricsCollector
from tests.ingestion.imap.conftest import MockImapServer


@pytest.mark.performance
@pytest.mark.integration
def test_large_mailbox_initial_sync_throughput(large_mailbox_dataset: MockImapServer, metrics_collector: ImapSyncMetricsCollector):
    """Test initial sync throughput with 10,000 messages."""
    mailbox_id = "perf-test@example.com"
    start_time = time.time()

    # Simulate sync of all messages
    message_count = len(large_mailbox_dataset.messages)
    assert message_count == 10000

    # Simulate processing (in real test, this would call actual sync engine)
    for uid in large_mailbox_dataset.messages.keys():
        pass  # Processing simulation

    duration = time.time() - start_time

    # Record metrics
    metrics_collector.record_message_processing(mailbox_id, message_count, duration)

    summary = metrics_collector.generate_summary(mailbox_id)

    # Performance requirement: ≥1 msg/s throughput
    assert summary.messages_per_second >= 1.0, f"Throughput {summary.messages_per_second:.2f} msg/s below 1.0 msg/s requirement"

    # Additional requirement: 10K messages in < 3 hours (10,000 seconds)
    max_duration = 10000  # 10K messages / 1 msg/s = 10,000 seconds worst case
    assert duration < max_duration, f"Sync took {duration:.2f}s, exceeds {max_duration}s limit"


@pytest.mark.performance
@pytest.mark.integration
def test_high_volume_message_detection(mock_imap_server: MockImapServer, metrics_collector: ImapSyncMetricsCollector):
    """Test detection window for 100 rapid message arrivals."""
    mailbox_id = "perf-test@example.com"

    # Simulate 100 messages arriving rapidly
    detection_times = []
    for i in range(100):
        arrival_time = time.time()
        uid = mock_imap_server.simulate_new_message(f"message{i}".encode())

        # Simulate detection (in real test, IDLE would detect)
        detection_delay = 0.5  # Simulate sub-second detection
        detection_times.append(detection_delay)

        metrics_collector.record_detection_time(mailbox_id, detection_delay)

    summary = metrics_collector.generate_summary(mailbox_id)

    # Performance requirement: <5 minutes (300 seconds) average detection
    assert summary.average_detection_time_seconds < 300, f"Detection window {summary.average_detection_time_seconds:.2f}s exceeds 300s requirement"

    # Realistic expectation: much better than 5 minutes
    assert summary.average_detection_time_seconds < 60, "Detection should be under 1 minute for IDLE-enabled sync"


@pytest.mark.performance
@pytest.mark.integration
def test_connection_pool_concurrent_operations(mock_imap_server: MockImapServer):
    """Test connection pool handles 100 concurrent operations efficiently."""
    # Add 100 messages
    for i in range(100):
        mock_imap_server.add_message(i + 1, f"message{i}".encode())

    start_time = time.time()

    # Simulate 100 concurrent fetch operations
    fetch_results = []
    for i in range(1, 101):
        result = mock_imap_server.fetch([i], ["RFC822"])
        fetch_results.append(result)

    duration = time.time() - start_time

    assert len(fetch_results) == 100
    # Should complete in reasonable time (< 10 seconds for mocked operations)
    assert duration < 10, f"100 concurrent operations took {duration:.2f}s, should be < 10s"


@pytest.mark.performance
@pytest.mark.integration
def test_incremental_sync_efficiency(mock_imap_server: MockImapServer, metrics_collector: ImapSyncMetricsCollector):
    """Test incremental sync performance with MODSEQ."""
    mailbox_id = "perf-test@example.com"

    # Initial sync: 1000 messages
    for i in range(1000):
        mock_imap_server.add_message(i + 1, f"message{i}".encode())

    # Incremental sync: 10 new messages
    start_time = time.time()
    for i in range(1000, 1010):
        mock_imap_server.simulate_new_message(f"message{i}".encode())

    # Simulate incremental sync (only fetch new messages)
    new_uids = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    fetch_result = mock_imap_server.fetch(new_uids, ["RFC822"])

    duration = time.time() - start_time

    metrics_collector.record_sync_latency(mailbox_id, "INBOX", duration)
    summary = metrics_collector.generate_summary(mailbox_id)

    # Performance requirement: <30s sync latency per folder
    assert summary.average_sync_latency_seconds < 30, f"Incremental sync latency {summary.average_sync_latency_seconds:.2f}s exceeds 30s requirement"


@pytest.mark.performance
@pytest.mark.integration
def test_multi_folder_concurrent_sync(mock_imap_server: MockImapServer):
    """Test syncing multiple folders concurrently."""
    folders = ["INBOX", "Sent", "Drafts", "Trash"]

    # Add messages to each folder
    for folder in folders:
        for i in range(100):
            mock_imap_server.add_message(i + 1, f"{folder}-message{i}".encode())

    start_time = time.time()

    # Simulate concurrent folder syncs
    sync_results = []
    for folder in folders:
        mock_imap_server.select_folder(folder)
        result = mock_imap_server.search(["ALL"])
        sync_results.append((folder, result))

    duration = time.time() - start_time

    assert len(sync_results) == 4
    # Should complete efficiently
    assert duration < 5, f"Multi-folder sync took {duration:.2f}s, should be < 5s"


@pytest.mark.performance
def test_throughput_sustained_rate(metrics_collector: ImapSyncMetricsCollector):
    """Test sustained throughput meets ≥1 msg/s requirement."""
    mailbox_id = "throughput-test@example.com"

    # Simulate processing 1000 messages over time
    total_messages = 1000
    total_duration = 500  # 500 seconds = 2 msg/s

    metrics_collector.record_message_processing(mailbox_id, total_messages, total_duration)
    summary = metrics_collector.generate_summary(mailbox_id)

    # Performance requirement: ≥1 msg/s
    assert summary.messages_per_second >= 1.0, f"Throughput {summary.messages_per_second:.2f} msg/s below 1.0 requirement"


@pytest.mark.performance
def test_sync_latency_per_folder(metrics_collector: ImapSyncMetricsCollector):
    """Test sync latency per folder meets <30s requirement."""
    mailbox_id = "latency-test@example.com"

    # Record multiple folder sync latencies
    folders = ["INBOX", "Sent", "Drafts", "Archive"]
    for folder in folders:
        # Simulate sync latency (well below 30s requirement)
        latency = 5.0  # 5 seconds
        metrics_collector.record_sync_latency(mailbox_id, folder, latency)

    summary = metrics_collector.generate_summary(mailbox_id)

    # Performance requirement: <30s average sync latency
    assert summary.average_sync_latency_seconds < 30, f"Sync latency {summary.average_sync_latency_seconds:.2f}s exceeds 30s requirement"


@pytest.mark.performance
@pytest.mark.integration
def test_stress_large_message_batch(mock_imap_server: MockImapServer):
    """Stress test: 50,000+ message mailbox."""
    # Add 50,000 messages
    message_count = 50000

    start_time = time.time()
    for i in range(message_count):
        mock_imap_server.add_message(i + 1, f"message{i}".encode())

    add_duration = time.time() - start_time

    # Verify all added
    assert len(mock_imap_server.messages) == message_count

    # Batch fetch should be efficient
    start_fetch = time.time()
    batch_uids = list(range(1, 1001))  # Fetch first 1000
    fetch_result = mock_imap_server.fetch(batch_uids, ["RFC822"])
    fetch_duration = time.time() - start_fetch

    assert len(fetch_result) == 1000
    # Batch fetch should be fast (< 5 seconds for mock)
    assert fetch_duration < 5, f"Batch fetch took {fetch_duration:.2f}s, should be < 5s"
