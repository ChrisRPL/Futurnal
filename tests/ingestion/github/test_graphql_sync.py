"""Tests for GitHub GraphQL sync implementation."""

import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile

from futurnal.ingestion.github.graphql_sync import GraphQLRepositorySync
from futurnal.ingestion.github.descriptor import (
    GitHubRepositoryDescriptor,
    SyncMode,
    VisibilityType,
    Provenance,
)
from futurnal.ingestion.github.sync_models import (
    SyncState,
    SyncStatus,
    SyncStrategy,
)


@pytest.fixture
def temp_workspace():
    """Create temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_api_client_manager():
    """Create mock API client manager."""
    manager = MagicMock()
    manager.graphql_request = AsyncMock()
    return manager


@pytest.fixture
def graphql_sync(mock_api_client_manager, temp_workspace):
    """Create GraphQL sync instance."""
    return GraphQLRepositorySync(
        api_client_manager=mock_api_client_manager,
        workspace_dir=temp_workspace,
    )


@pytest.fixture
def test_descriptor():
    """Create test repository descriptor."""
    return GitHubRepositoryDescriptor(
        id="test-repo-id",
        owner="testowner",
        repo="testrepo",
        full_name="testowner/testrepo",
        visibility=VisibilityType.PUBLIC,
        credential_id="test-cred",
        sync_mode=SyncMode.GRAPHQL_API,
        provenance=Provenance(
            os_user="test",
            machine_id_hash="test123",
            tool_version="1.0.0",
        ),
    )


@pytest.fixture
def test_strategy():
    """Create test sync strategy."""
    return SyncStrategy(
        branches=["main"],
        file_patterns=[],
        exclude_patterns=[".git/", "*.pyc"],
        max_file_size_mb=10,
        fetch_file_content=True,
        batch_size=10,
    )


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


def test_graphql_sync_initialization(mock_api_client_manager, temp_workspace):
    """Test GraphQL sync initialization."""
    sync = GraphQLRepositorySync(
        api_client_manager=mock_api_client_manager,
        workspace_dir=temp_workspace,
    )

    assert sync.api_client_manager == mock_api_client_manager
    assert sync.workspace_dir == temp_workspace
    assert temp_workspace.exists()


def test_graphql_sync_default_workspace():
    """Test GraphQL sync with default workspace."""
    mock_manager = MagicMock()
    sync = GraphQLRepositorySync(api_client_manager=mock_manager)

    expected_workspace = (
        Path.home() / ".futurnal" / "workspace" / "github" / "graphql"
    )
    assert sync.workspace_dir == expected_workspace


# ---------------------------------------------------------------------------
# Tree Fetching Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_repository_tree_basic(graphql_sync, test_descriptor):
    """Test fetching repository tree."""
    # Mock GraphQL response
    mock_response = {
        "data": {
            "repository": {
                "object": {
                    "entries": [
                        {
                            "name": "README.md",
                            "type": "blob",
                            "mode": "100644",
                            "path": "README.md",
                            "object": {
                                "byteSize": 1024,
                                "isBinary": False,
                                "oid": "abc123",
                            },
                        },
                        {
                            "name": "main.py",
                            "type": "blob",
                            "mode": "100644",
                            "path": "src/main.py",
                            "object": {
                                "byteSize": 2048,
                                "isBinary": False,
                                "oid": "def456",
                            },
                        },
                    ]
                }
            }
        }
    }

    graphql_sync.api_client_manager.graphql_request.return_value = mock_response

    entries = await graphql_sync._fetch_repository_tree(
        descriptor=test_descriptor,
        branch="main",
        path="",
    )

    assert len(entries) == 2
    assert entries[0].name == "README.md"
    assert entries[0].size == 1024
    assert entries[1].path == "src/main.py"


@pytest.mark.asyncio
async def test_fetch_repository_tree_empty(graphql_sync, test_descriptor):
    """Test fetching empty repository tree."""
    mock_response = {
        "data": {
            "repository": {
                "object": {
                    "entries": []
                }
            }
        }
    }

    graphql_sync.api_client_manager.graphql_request.return_value = mock_response

    entries = await graphql_sync._fetch_repository_tree(
        descriptor=test_descriptor,
        branch="main",
    )

    assert len(entries) == 0


# ---------------------------------------------------------------------------
# Commit SHA Fetching Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_latest_commit_sha(graphql_sync, test_descriptor):
    """Test fetching latest commit SHA."""
    mock_response = {
        "data": {
            "repository": {
                "ref": {
                    "target": {
                        "oid": "abc123def456"
                    }
                }
            }
        }
    }

    graphql_sync.api_client_manager.graphql_request.return_value = mock_response

    sha = await graphql_sync._fetch_latest_commit_sha(
        descriptor=test_descriptor,
        branch="main",
    )

    assert sha == "abc123def456"


# ---------------------------------------------------------------------------
# File Filtering Tests
# ---------------------------------------------------------------------------


def test_filter_files_no_patterns(graphql_sync, test_strategy):
    """Test filtering files with no patterns."""
    from futurnal.ingestion.github.sync_models import FileEntry

    entries = [
        FileEntry(path="src/main.py", name="main.py", type="blob", size=1024),
        FileEntry(path="README.md", name="README.md", type="blob", size=512),
        FileEntry(path="tests/test.py", name="test.py", type="blob", size=256),
    ]

    # No include patterns means include all (except excludes)
    filtered = graphql_sync._filter_files(entries, test_strategy)

    assert len(filtered) == 3


def test_filter_files_with_exclude(graphql_sync, test_strategy):
    """Test filtering with exclude patterns."""
    from futurnal.ingestion.github.sync_models import FileEntry

    entries = [
        FileEntry(path="src/main.py", name="main.py", type="blob", size=1024),
        FileEntry(path="cache.pyc", name="cache.pyc", type="blob", size=512),
        FileEntry(path=".git/config", name="config", type="blob", size=256),
    ]

    filtered = graphql_sync._filter_files(entries, test_strategy)

    # Should exclude .pyc and .git/
    assert len(filtered) == 1
    assert filtered[0].path == "src/main.py"


def test_filter_files_binary(graphql_sync, test_strategy):
    """Test filtering binary files."""
    from futurnal.ingestion.github.sync_models import FileEntry

    entries = [
        FileEntry(
            path="text.txt", name="text.txt", type="blob", size=1024, is_binary=False
        ),
        FileEntry(
            path="binary.bin", name="binary.bin", type="blob", size=1024, is_binary=True
        ),
    ]

    filtered = graphql_sync._filter_files(entries, test_strategy)

    # Should exclude binary
    assert len(filtered) == 1
    assert filtered[0].path == "text.txt"


def test_filter_files_too_large(graphql_sync, test_strategy):
    """Test filtering files that are too large."""
    from futurnal.ingestion.github.sync_models import FileEntry

    entries = [
        FileEntry(path="small.txt", name="small.txt", type="blob", size=1024),
        FileEntry(
            path="large.txt",
            name="large.txt",
            type="blob",
            size=20 * 1024 * 1024,  # 20 MB
        ),
    ]

    filtered = graphql_sync._filter_files(entries, test_strategy)

    # Should exclude large file (max is 10 MB)
    assert len(filtered) == 1
    assert filtered[0].path == "small.txt"


# ---------------------------------------------------------------------------
# Batch File Content Fetching Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_file_contents_batch(graphql_sync, test_descriptor):
    """Test fetching file contents in batches."""
    from futurnal.ingestion.github.sync_models import FileEntry

    files = [
        FileEntry(path="file1.txt", name="file1.txt", type="blob", size=100),
        FileEntry(path="file2.txt", name="file2.txt", type="blob", size=200),
    ]

    mock_response = {
        "data": {
            "repository": {
                "file0": {
                    "text": "content1",
                    "byteSize": 100,
                    "isBinary": False,
                    "oid": "sha1",
                },
                "file1": {
                    "text": "content2",
                    "byteSize": 200,
                    "isBinary": False,
                    "oid": "sha2",
                },
            }
        }
    }

    graphql_sync.api_client_manager.graphql_request.return_value = mock_response

    contents = await graphql_sync._fetch_file_contents_batch(
        descriptor=test_descriptor,
        branch="main",
        files=files,
        batch_size=10,
    )

    assert len(contents) == 2
    assert contents[0].content == "content1"
    assert contents[1].content == "content2"


# ---------------------------------------------------------------------------
# Full Sync Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_repository_success(graphql_sync, test_descriptor, test_strategy):
    """Test successful repository sync."""
    # Mock tree response
    tree_response = {
        "data": {
            "repository": {
                "object": {
                    "entries": [
                        {
                            "name": "README.md",
                            "type": "blob",
                            "mode": "100644",
                            "path": "README.md",
                            "object": {
                                "byteSize": 1024,
                                "isBinary": False,
                                "oid": "abc123",
                            },
                        }
                    ]
                }
            }
        }
    }

    # Mock commit SHA response
    commit_response = {
        "data": {
            "repository": {
                "ref": {
                    "target": {
                        "oid": "commit123"
                    }
                }
            }
        }
    }

    # Mock file content response
    content_response = {
        "data": {
            "repository": {
                "file0": {
                    "text": "# Test README\nThis is test content",
                    "byteSize": 13,  # Match actual content length
                    "isBinary": False,
                    "oid": "abc123",
                }
            }
        }
    }

    # Configure mock to return different responses in order:
    # 1. tree fetch, 2. content fetch, 3. commit SHA fetch
    graphql_sync.api_client_manager.graphql_request.side_effect = [
        tree_response,
        content_response,
        commit_response,
    ]

    result = await graphql_sync.sync_repository(
        descriptor=test_descriptor,
        strategy=test_strategy,
    )

    assert result.status == SyncStatus.COMPLETED
    assert result.files_synced == 1
    assert result.files_failed == 0
    assert result.bytes_synced == 13


@pytest.mark.asyncio
async def test_sync_repository_with_errors(graphql_sync, test_descriptor, test_strategy):
    """Test sync with API errors."""
    # Mock to raise exception
    graphql_sync.api_client_manager.graphql_request.side_effect = Exception(
        "API Error"
    )

    result = await graphql_sync.sync_repository(
        descriptor=test_descriptor,
        strategy=test_strategy,
    )

    assert result.status == SyncStatus.FAILED
    assert result.error_message == "API Error"


@pytest.mark.asyncio
async def test_sync_repository_metadata_only(graphql_sync, test_descriptor):
    """Test sync with metadata only (no content fetch)."""
    strategy = SyncStrategy(
        branches=["main"],
        fetch_file_content=False,  # Metadata only
    )

    tree_response = {
        "data": {
            "repository": {
                "object": {
                    "entries": [
                        {
                            "name": "README.md",
                            "type": "blob",
                            "mode": "100644",
                            "path": "README.md",
                            "object": {
                                "byteSize": 1024,
                                "isBinary": False,
                                "oid": "abc123",
                            },
                        }
                    ]
                }
            }
        }
    }

    commit_response = {
        "data": {
            "repository": {
                "ref": {
                    "target": {
                        "oid": "commit123"
                    }
                }
            }
        }
    }

    graphql_sync.api_client_manager.graphql_request.side_effect = [
        tree_response,
        commit_response,
    ]

    result = await graphql_sync.sync_repository(
        descriptor=test_descriptor,
        strategy=strategy,
    )

    assert result.status == SyncStatus.COMPLETED
    assert result.files_synced == 1
    assert result.bytes_synced == 0  # No content fetched
