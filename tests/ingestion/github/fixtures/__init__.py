"""Test fixtures for GitHub connector testing.

This module provides reusable fixtures for testing the GitHub connector,
including mock repositories of different sizes, enhanced GitHub API mocks,
and test data generators.
"""

from tests.ingestion.github.fixtures.repositories import (
    large_test_repo_fixture,
    medium_test_repo_fixture,
    repo_with_force_push,
    repo_with_multiple_branches,
    small_test_repo_fixture,
)
from tests.ingestion.github.fixtures.mock_github_enhanced import (
    EnhancedMockGitHubAPI,
    circuit_breaker_open_api,
    enhanced_mock_github_api,
    rate_limit_exhausted_api,
)

__all__ = [
    # Repository fixtures
    "small_test_repo_fixture",
    "medium_test_repo_fixture",
    "large_test_repo_fixture",
    "repo_with_force_push",
    "repo_with_multiple_branches",
    # Enhanced API mocks
    "EnhancedMockGitHubAPI",
    "enhanced_mock_github_api",
    "rate_limit_exhausted_api",
    "circuit_breaker_open_api",
]
