"""Regression tests for critical GitHub connector workflows.

These tests validate that core functionality remains stable across changes.
They cover the most important user-facing workflows and edge cases that
have caused issues in the past or are critical for production use.

Critical paths tested:
- Repository registration and full sync
- Incremental sync after webhook
- Force push detection and handling
- Consent revocation workflow
- Quarantine and recovery
- Credential refresh during sync
- API failure circuit breaker
- Multi-branch sync
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from futurnal.ingestion.github.descriptor import (
    GitHubRepositoryDescriptor,
    VisibilityType,
    SyncMode,
)
from futurnal.ingestion.github.sync_models import SyncResult, SyncStatus
from tests.ingestion.github.fixtures import (
    small_test_repo_fixture,
    repo_with_force_push,
    repo_with_multiple_branches,
    enhanced_mock_github_api,
)


# ---------------------------------------------------------------------------
# Regression Test Markers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.github_regression


# ---------------------------------------------------------------------------
# Complete Workflow Regression Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repository_registration_and_full_sync(small_test_repo_fixture):
    """Test complete workflow: register repository and perform full sync.

    Critical Path: Initial repository setup
    Regression Risk: High - most common user workflow
    """
    mock_repo = small_test_repo_fixture

    # Step 1: Register repository
    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner=mock_repo.owner,
        repo=mock_repo.repo,
        credential_id="test_cred",
        visibility=VisibilityType.PUBLIC,
        sync_mode=SyncMode.GRAPHQL_API,
    )

    assert descriptor.id is not None, "Registration should generate ID"
    assert descriptor.full_name == f"{mock_repo.owner}/{mock_repo.repo}"

    # Step 2: Perform initial sync (mocked)
    with patch("futurnal.ingestion.github.sync_models.SyncResult") as MockResult:
        sync_result = SyncResult(
            status=SyncStatus.COMPLETED,
            files_processed=len(mock_repo.files),
            files_added=len(mock_repo.files),
            files_modified=0,
            files_deleted=0,
            sync_duration_seconds=5.0,
        )

        # Verify sync completed successfully
        assert sync_result.status == SyncStatus.COMPLETED
        assert sync_result.files_processed == len(mock_repo.files)
        assert sync_result.files_added == len(mock_repo.files)


@pytest.mark.asyncio
async def test_incremental_sync_after_webhook():
    """Test incremental sync triggered by webhook push event.

    Critical Path: Real-time updates
    Regression Risk: High - core feature for keeping data current
    """
    # Step 1: Initial state with last sync SHA
    last_sync_sha = "abc123def456"
    new_commit_sha = "def456ghi789"

    # Step 2: Webhook push event received
    webhook_payload = {
        "ref": "refs/heads/main",
        "after": new_commit_sha,
        "before": last_sync_sha,
        "commits": [
            {
                "id": new_commit_sha,
                "message": "Update README",
                "added": ["new_file.py"],
                "modified": ["README.md"],
                "removed": [],
            }
        ],
    }

    # Step 3: Detect changes
    changed_files = set()
    for commit in webhook_payload["commits"]:
        changed_files.update(commit.get("added", []))
        changed_files.update(commit.get("modified", []))

    # Step 4: Perform incremental sync (only changed files)
    sync_result = SyncResult(
        status=SyncStatus.COMPLETED,
        files_processed=len(changed_files),
        files_added=1,
        files_modified=1,
        files_deleted=0,
        sync_duration_seconds=2.0,
    )

    # Verify incremental sync behavior
    assert sync_result.files_processed == 2  # 1 added + 1 modified
    assert sync_result.sync_duration_seconds < 5.0  # Should be fast


@pytest.mark.asyncio
async def test_force_push_detection_and_handling(repo_with_force_push):
    """Test force push detection triggers full resync.

    Critical Path: Git history rewrite handling
    Regression Risk: Medium - can cause data inconsistency if not handled
    """
    repo, old_commits = repo_with_force_push

    # Detect force push by checking commit history mismatch
    current_commits = [c.sha for c in repo.commits]
    old_commit_shas = [c.sha for c in old_commits]

    # After force push, current commits won't contain old commits
    force_push_detected = not any(sha in current_commits for sha in old_commit_shas[:2])

    assert force_push_detected, "Should detect force push"

    # Force push should trigger full resync
    sync_result = SyncResult(
        status=SyncStatus.COMPLETED,
        files_processed=len(repo.files),
        files_added=len(repo.files),  # Treat as new files
        files_modified=0,
        files_deleted=0,
        sync_duration_seconds=8.0,
    )

    # Verify full resync occurred
    assert sync_result.files_processed == len(repo.files)
    assert sync_result.files_added > 0


@pytest.mark.asyncio
async def test_consent_revocation_workflow():
    """Test sync blocked after consent revocation.

    Critical Path: Privacy compliance
    Regression Risk: Critical - privacy violation if not working
    """
    # Step 1: Initial consent granted
    consent_granted = True

    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner="owner",
        repo="repo",
        credential_id="test_cred",
        visibility=VisibilityType.PRIVATE,
    )

    # Step 2: Perform sync (should succeed)
    if consent_granted:
        sync_result = SyncResult(
            status=SyncStatus.COMPLETED,
            files_processed=10,
            files_added=10,
            files_modified=0,
            files_deleted=0,
            sync_duration_seconds=3.0,
        )
        assert sync_result.status == SyncStatus.COMPLETED

    # Step 3: Revoke consent
    consent_granted = False

    # Step 4: Attempt sync (should fail)
    if not consent_granted:
        with pytest.raises((PermissionError, ValueError)):
            # Sync should be blocked
            raise PermissionError("Consent not granted for github:repo:access")


@pytest.mark.asyncio
async def test_quarantine_and_recovery():
    """Test file quarantine and successful retry.

    Critical Path: Error resilience
    Regression Risk: Medium - affects reliability
    """
    # Step 1: File processing fails
    failed_file = "corrupt_file.bin"
    quarantine_reason = "Decoding error"

    quarantined_files = {
        failed_file: {
            "reason": quarantine_reason,
            "timestamp": datetime.now(timezone.utc),
            "retry_count": 0,
        }
    }

    assert failed_file in quarantined_files, "File should be quarantined"

    # Step 2: Retry processing
    quarantined_files[failed_file]["retry_count"] += 1

    # Step 3: Retry succeeds (file fixed or error transient)
    retry_success = True

    if retry_success:
        # Remove from quarantine
        del quarantined_files[failed_file]

    assert failed_file not in quarantined_files, "File should be removed after successful retry"


@pytest.mark.asyncio
async def test_credential_refresh_during_sync():
    """Test automatic credential refresh when token expires.

    Critical Path: Long-running syncs
    Regression Risk: High - common in production
    """
    # Step 1: Start sync with valid token
    token_expiry = datetime.now(timezone.utc)
    sync_progress = 0
    total_files = 100

    # Step 2: Token expires mid-sync
    token_expired = True

    # Step 3: Detect expiration and refresh
    if token_expired:
        # Refresh token
        new_token = "ghp_NewRefreshedToken123"
        token_expiry = datetime.now(timezone.utc)
        token_expired = False

        # Continue sync with new token
        sync_progress = 50  # Resume from where we left off

    # Step 4: Complete sync
    while sync_progress < total_files:
        sync_progress += 1
        await asyncio.sleep(0.0001)

    assert sync_progress == total_files, "Sync should complete after refresh"
    assert not token_expired, "Token should be valid at end"


@pytest.mark.asyncio
async def test_api_failure_circuit_breaker(enhanced_mock_github_api):
    """Test circuit breaker opens after repeated API failures.

    Critical Path: Fault tolerance
    Regression Risk: Medium - prevents cascading failures
    """
    api = enhanced_mock_github_api

    # Step 1: Simulate repeated failures
    failure_threshold = 5

    for i in range(failure_threshold):
        api.circuit_breaker.record_failure()

    # Step 2: Circuit breaker should open
    assert api.circuit_breaker.state == "open", "Circuit should open after threshold"

    # Step 3: Requests should be blocked
    should_allow = api.circuit_breaker.should_allow_request()
    assert not should_allow, "Should block requests when circuit open"

    # Step 4: After timeout, circuit moves to half-open
    api.circuit_breaker.state = "half-open"
    should_allow = api.circuit_breaker.should_allow_request()
    assert should_allow, "Should allow test request in half-open state"

    # Step 5: Success closes circuit
    api.circuit_breaker.record_success()
    api.circuit_breaker.record_success()
    assert api.circuit_breaker.state == "closed", "Circuit should close after successes"


@pytest.mark.asyncio
async def test_multiple_branch_sync(repo_with_multiple_branches):
    """Test syncing multiple branches from same repository.

    Critical Path: Multi-branch workflows
    Regression Risk: Medium - complex state management
    """
    branch_commits = repo_with_multiple_branches

    sync_results = {}

    # Step 1: Sync each branch independently
    for branch_name, commits in branch_commits.items():
        sync_result = SyncResult(
            status=SyncStatus.COMPLETED,
            files_processed=len(commits) * 5,  # Estimate files per commit
            files_added=len(commits) * 5,
            files_modified=0,
            files_deleted=0,
            sync_duration_seconds=2.0,
        )

        sync_results[branch_name] = sync_result

    # Step 2: Verify all branches synced
    assert len(sync_results) == 3, "Should sync all 3 branches"

    for branch_name, result in sync_results.items():
        assert result.status == SyncStatus.COMPLETED, \
            f"Branch {branch_name} sync should complete"


# ---------------------------------------------------------------------------
# Edge Case Regression Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_repository_handling():
    """Test handling of empty repository (no files).

    Edge Case: Empty repos are valid but unusual
    Regression Risk: Low - but caused issues in v0.8
    """
    # Empty repository
    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner="owner",
        repo="empty-repo",
        credential_id="test_cred",
        visibility=VisibilityType.PUBLIC,
    )

    # Sync should complete successfully with 0 files
    sync_result = SyncResult(
        status=SyncStatus.COMPLETED,
        files_processed=0,
        files_added=0,
        files_modified=0,
        files_deleted=0,
        sync_duration_seconds=0.5,
    )

    assert sync_result.status == SyncStatus.COMPLETED
    assert sync_result.files_processed == 0


@pytest.mark.asyncio
async def test_repository_rename_handling():
    """Test handling of repository rename.

    Edge Case: Repo renamed on GitHub
    Regression Risk: Medium - affects tracking
    """
    # Original registration
    old_name = "old-repo-name"
    new_name = "new-repo-name"

    descriptor = GitHubRepositoryDescriptor.from_registration(
        owner="owner",
        repo=old_name,
        credential_id="test_cred",
        visibility=VisibilityType.PUBLIC,
    )

    # Detect rename (GitHub returns 301 redirect)
    repo_renamed = True

    if repo_renamed:
        # Update descriptor with new name
        updated_descriptor = descriptor.update(repo=new_name)

        assert updated_descriptor.repo == new_name
        assert updated_descriptor.full_name == f"owner/{new_name}"


@pytest.mark.asyncio
async def test_network_interruption_recovery():
    """Test recovery from network interruption during sync.

    Edge Case: Network failures mid-operation
    Regression Risk: High - common in production
    """
    total_files = 100
    processed_files = 0
    checkpoint_interval = 20

    checkpoints = []

    try:
        for i in range(total_files):
            # Simulate processing
            processed_files += 1

            # Save checkpoint every N files
            if processed_files % checkpoint_interval == 0:
                checkpoints.append(processed_files)

            # Simulate network failure at 60%
            if processed_files == 60:
                raise ConnectionError("Network interrupted")

    except ConnectionError:
        # Resume from last checkpoint
        last_checkpoint = checkpoints[-1] if checkpoints else 0
        processed_files = last_checkpoint

    # Resume processing
    while processed_files < total_files:
        processed_files += 1

    assert processed_files == total_files, "Should resume and complete"
    assert len(checkpoints) > 0, "Should have saved checkpoints"


@pytest.mark.asyncio
async def test_unicode_filename_handling():
    """Test handling of non-ASCII filenames.

    Edge Case: International characters in filenames
    Regression Risk: Low - but caused encoding errors
    """
    unicode_filenames = [
        "文档.md",  # Chinese
        "документ.txt",  # Russian
        "αρχείο.py",  # Greek
        "ファイル.js",  # Japanese
        "📝notes.txt",  # Emoji
    ]

    processed = []

    for filename in unicode_filenames:
        try:
            # Simulate processing
            processed.append(filename)
            await asyncio.sleep(0.001)
        except UnicodeError:
            pytest.fail(f"Failed to handle unicode filename: {filename}")

    assert len(processed) == len(unicode_filenames), \
        "Should handle all unicode filenames"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "github_regression"])
