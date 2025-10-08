"""Test repository fixtures for GitHub connector testing.

Provides mock repositories of different sizes (small, medium, large) with
realistic file structures, commit histories, and metadata for comprehensive
testing of the GitHub connector.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Repository Data Models
# ---------------------------------------------------------------------------


class MockFile:
    """Represents a file in a mock repository."""

    def __init__(
        self,
        path: str,
        content: str,
        size: int,
        sha: str,
        language: Optional[str] = None,
    ):
        self.path = path
        self.content = content
        self.size = size
        self.sha = sha
        self.language = language


class MockCommit:
    """Represents a commit in a mock repository."""

    def __init__(
        self,
        sha: str,
        message: str,
        author: str,
        timestamp: datetime,
        files_changed: List[str],
    ):
        self.sha = sha
        self.message = message
        self.author = author
        self.timestamp = timestamp
        self.files_changed = files_changed


class MockRepository:
    """Represents a complete mock repository for testing."""

    def __init__(
        self,
        owner: str,
        repo: str,
        description: str,
        visibility: str = "public",
        default_branch: str = "main",
    ):
        self.owner = owner
        self.repo = repo
        self.full_name = f"{owner}/{repo}"
        self.description = description
        self.visibility = visibility
        self.default_branch = default_branch
        self.files: List[MockFile] = []
        self.commits: List[MockCommit] = []
        self.branches: List[str] = [default_branch]
        self.size_kb = 0

    def add_file(self, file: MockFile):
        """Add a file to the repository."""
        self.files.append(file)
        self.size_kb += file.size // 1024

    def add_commit(self, commit: MockCommit):
        """Add a commit to the repository."""
        self.commits.append(commit)

    def get_file_tree(self) -> Dict[str, Any]:
        """Get file tree structure."""
        tree = {}
        for file in self.files:
            parts = Path(file.path).parts
            current = tree
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = {
                "sha": file.sha,
                "size": file.size,
                "language": file.language,
            }
        return tree

    def to_github_api_response(self) -> Dict[str, Any]:
        """Convert to GitHub API repository response format."""
        now = datetime.now(timezone.utc)
        return {
            "id": hash(self.full_name) & 0xFFFFFFFF,
            "name": self.repo,
            "full_name": self.full_name,
            "owner": {
                "login": self.owner,
                "id": hash(self.owner) & 0xFFFFFFFF,
                "type": "User",
            },
            "description": self.description,
            "private": self.visibility == "private",
            "visibility": self.visibility,
            "fork": False,
            "created_at": (now - timedelta(days=365)).isoformat(),
            "updated_at": (now - timedelta(days=1)).isoformat(),
            "pushed_at": (now - timedelta(hours=2)).isoformat(),
            "size": self.size_kb,
            "language": self._detect_primary_language(),
            "default_branch": self.default_branch,
            "has_issues": True,
            "has_wiki": True,
            "has_pages": False,
            "archived": False,
        }

    def _detect_primary_language(self) -> Optional[str]:
        """Detect primary language from file extensions."""
        language_counts = {}
        for file in self.files:
            if file.language:
                language_counts[file.language] = (
                    language_counts.get(file.language, 0) + 1
                )
        if language_counts:
            return max(language_counts, key=language_counts.get)
        return None


# ---------------------------------------------------------------------------
# File Generators
# ---------------------------------------------------------------------------


def generate_python_file(path: str, sha: str) -> MockFile:
    """Generate a Python source file."""
    content = f'''"""Module: {path}"""

import os
from typing import Any, Dict, List


def example_function() -> str:
    """Example function for testing."""
    return "Hello, World!"


class ExampleClass:
    """Example class for testing."""

    def __init__(self, name: str):
        self.name = name

    def greet(self) -> str:
        return f"Hello, {{self.name}}!"


if __name__ == "__main__":
    print(example_function())
'''
    return MockFile(path, content, len(content.encode()), sha, "Python")


def generate_markdown_file(path: str, sha: str) -> MockFile:
    """Generate a Markdown documentation file."""
    content = f"""# {Path(path).stem}

This is documentation for testing purposes.

## Features

- Feature 1
- Feature 2
- Feature 3

## Usage

```python
from example import example_function
result = example_function()
```

## License

MIT License
"""
    return MockFile(path, content, len(content.encode()), sha, "Markdown")


def generate_json_file(path: str, sha: str) -> MockFile:
    """Generate a JSON configuration file."""
    content = """{
  "name": "test-package",
  "version": "1.0.0",
  "description": "Test package for GitHub connector",
  "main": "index.js",
  "scripts": {
    "test": "jest",
    "build": "webpack"
  },
  "dependencies": {
    "react": "^18.0.0",
    "lodash": "^4.17.21"
  }
}
"""
    return MockFile(path, content, len(content.encode()), sha, "JSON")


def generate_javascript_file(path: str, sha: str) -> MockFile:
    """Generate a JavaScript source file."""
    content = f'''// {path}

export function exampleFunction() {{
  return "Hello, World!";
}}

export class ExampleClass {{
  constructor(name) {{
    this.name = name;
  }}

  greet() {{
    return `Hello, ${{this.name}}!`;
  }}
}}

export default {{
  exampleFunction,
  ExampleClass
}};
'''
    return MockFile(path, content, len(content.encode()), sha, "JavaScript")


def generate_yaml_file(path: str, sha: str) -> MockFile:
    """Generate a YAML configuration file."""
    content = f"""# {path}
name: Test Workflow
on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: npm test
"""
    return MockFile(path, content, len(content.encode()), sha, "YAML")


# ---------------------------------------------------------------------------
# Small Repository Fixture (~10 files)
# ---------------------------------------------------------------------------


@pytest.fixture
def small_test_repo_fixture() -> MockRepository:
    """Small test repository with ~10 files for basic testing.

    Target sync time: <10 seconds
    Use case: Quick tests, basic workflow validation
    """
    repo = MockRepository(
        owner="octocat",
        repo="Hello-World",
        description="Small test repository for GitHub connector testing",
        visibility="public",
        default_branch="main",
    )

    # Add basic project structure
    files = [
        generate_python_file("src/main.py", "sha_main_001"),
        generate_python_file("src/utils.py", "sha_utils_001"),
        generate_python_file("tests/test_main.py", "sha_test_001"),
        generate_markdown_file("README.md", "sha_readme_001"),
        generate_markdown_file("CONTRIBUTING.md", "sha_contrib_001"),
        generate_json_file("package.json", "sha_package_001"),
        generate_yaml_file(".github/workflows/test.yml", "sha_workflow_001"),
        MockFile(
            ".gitignore",
            "*.pyc\n__pycache__/\n.env\nnode_modules/\n",
            50,
            "sha_gitignore_001",
        ),
        MockFile("LICENSE", "MIT License\n\nCopyright (c) 2024", 30, "sha_license_001"),
        MockFile(
            "requirements.txt",
            "pytest>=7.0.0\nrequests>=2.28.0\n",
            40,
            "sha_requirements_001",
        ),
    ]

    for file in files:
        repo.add_file(file)

    # Add commit history
    now = datetime.now(timezone.utc)
    commits = [
        MockCommit(
            "abc123def456",
            "Initial commit",
            "octocat",
            now - timedelta(days=30),
            ["README.md", "LICENSE"],
        ),
        MockCommit(
            "def456ghi789",
            "Add main source code",
            "octocat",
            now - timedelta(days=25),
            ["src/main.py", "src/utils.py"],
        ),
        MockCommit(
            "ghi789jkl012",
            "Add tests",
            "octocat",
            now - timedelta(days=20),
            ["tests/test_main.py"],
        ),
        MockCommit(
            "jkl012mno345",
            "Add CI/CD workflow",
            "octocat",
            now - timedelta(days=10),
            [".github/workflows/test.yml"],
        ),
        MockCommit(
            "mno345pqr678",
            "Update documentation",
            "octocat",
            now - timedelta(days=2),
            ["README.md", "CONTRIBUTING.md"],
        ),
    ]

    for commit in commits:
        repo.add_commit(commit)

    return repo


# ---------------------------------------------------------------------------
# Medium Repository Fixture (~500 files)
# ---------------------------------------------------------------------------


@pytest.fixture
def medium_test_repo_fixture() -> MockRepository:
    """Medium test repository with ~500 files for realistic testing.

    Target sync time: <60 seconds
    Target API requests: <100
    Use case: Performance testing, realistic workflows
    """
    repo = MockRepository(
        owner="test-org",
        repo="medium-project",
        description="Medium-sized test repository with realistic structure",
        visibility="private",
        default_branch="main",
    )

    # Add branches
    repo.branches = ["main", "develop", "feature/new-feature"]

    file_counter = 0

    # Generate source code files (300 files)
    for i in range(50):
        # Python module with tests
        repo.add_file(
            generate_python_file(
                f"src/modules/module_{i:03d}/core.py", f"sha_py_{file_counter:05d}"
            )
        )
        file_counter += 1
        repo.add_file(
            generate_python_file(
                f"src/modules/module_{i:03d}/utils.py", f"sha_py_{file_counter:05d}"
            )
        )
        file_counter += 1
        repo.add_file(
            generate_python_file(
                f"src/modules/module_{i:03d}/models.py", f"sha_py_{file_counter:05d}"
            )
        )
        file_counter += 1
        repo.add_file(
            generate_python_file(
                f"tests/modules/test_module_{i:03d}.py",
                f"sha_test_{file_counter:05d}",
            )
        )
        file_counter += 1

        # JavaScript components
        repo.add_file(
            generate_javascript_file(
                f"frontend/components/Component_{i:03d}.js",
                f"sha_js_{file_counter:05d}",
            )
        )
        file_counter += 1
        repo.add_file(
            generate_javascript_file(
                f"frontend/utils/util_{i:03d}.js", f"sha_js_{file_counter:05d}"
            )
        )
        file_counter += 1

    # Generate documentation files (100 files)
    for i in range(50):
        repo.add_file(
            generate_markdown_file(
                f"docs/modules/module_{i:03d}.md", f"sha_md_{file_counter:05d}"
            )
        )
        file_counter += 1
        repo.add_file(
            generate_markdown_file(
                f"docs/api/api_{i:03d}.md", f"sha_md_{file_counter:05d}"
            )
        )
        file_counter += 1

    # Generate configuration files (100 files)
    for i in range(50):
        repo.add_file(
            generate_json_file(
                f"config/environments/env_{i:03d}.json", f"sha_json_{file_counter:05d}"
            )
        )
        file_counter += 1
        repo.add_file(
            generate_yaml_file(
                f"config/workflows/workflow_{i:03d}.yml", f"sha_yaml_{file_counter:05d}"
            )
        )
        file_counter += 1

    # Add root files
    root_files = [
        generate_markdown_file("README.md", "sha_readme_medium"),
        generate_markdown_file("CONTRIBUTING.md", "sha_contrib_medium"),
        generate_markdown_file("CHANGELOG.md", "sha_changelog_medium"),
        generate_json_file("package.json", "sha_package_medium"),
        generate_json_file("tsconfig.json", "sha_tsconfig_medium"),
        MockFile(".gitignore", "*.pyc\n__pycache__/\n.env\n", 100, "sha_gitignore_medium"),
        MockFile("LICENSE", "MIT License", 50, "sha_license_medium"),
        MockFile("requirements.txt", "pytest>=7.0.0\n", 200, "sha_requirements_medium"),
    ]

    for file in root_files:
        repo.add_file(file)

    # Generate realistic commit history
    now = datetime.now(timezone.utc)
    for i in range(100):
        days_ago = 180 - i
        repo.add_commit(
            MockCommit(
                f"commit_sha_{i:05d}",
                f"Commit message {i}: Add feature or fix bug",
                "developer" if i % 3 == 0 else "contributor",
                now - timedelta(days=days_ago),
                [f"src/modules/module_{i % 50:03d}/core.py"],
            )
        )

    return repo


# ---------------------------------------------------------------------------
# Large Repository Fixture (~10k files, metadata only)
# ---------------------------------------------------------------------------


@pytest.fixture
def large_test_repo_fixture() -> MockRepository:
    """Large test repository with 10k+ files (metadata only for load testing).

    Use case: Load testing, stress testing, large-scale operations
    Note: Files contain minimal content to avoid memory issues
    """
    repo = MockRepository(
        owner="enterprise-org",
        repo="monorepo",
        description="Large monorepo for load and stress testing",
        visibility="private",
        default_branch="main",
    )

    # Add multiple branches
    repo.branches = ["main", "develop", "staging", "release/v1.0", "release/v2.0"]

    file_counter = 0

    # Generate large file structure (minimal content)
    # 1000 Python modules with 10 files each = 10,000 files
    for module in range(1000):
        for file_type in ["core", "models", "utils", "views", "tests", "types", "schemas", "api", "db", "cache"]:
            path = f"services/service_{module:04d}/{file_type}.py"
            # Minimal content to save memory
            content = f"# {path}\n# Auto-generated for testing\n"
            repo.add_file(
                MockFile(
                    path,
                    content,
                    len(content.encode()),
                    f"sha_{file_counter:08d}",
                    "Python",
                )
            )
            file_counter += 1

    # Add root files
    root_files = [
        MockFile("README.md", "# Large Monorepo", 1000, "sha_readme_large"),
        MockFile(".gitignore", "*.pyc\n", 500, "sha_gitignore_large"),
        MockFile("LICENSE", "MIT", 50, "sha_license_large"),
    ]

    for file in root_files:
        repo.add_file(file)

    # Generate extensive commit history (1000 commits)
    now = datetime.now(timezone.utc)
    for i in range(1000):
        hours_ago = 10000 - i
        repo.add_commit(
            MockCommit(
                f"large_commit_{i:06d}",
                f"Commit {i}: Development work",
                f"dev_{i % 50}",
                now - timedelta(hours=hours_ago),
                [f"services/service_{i % 1000:04d}/core.py"],
            )
        )

    return repo


# ---------------------------------------------------------------------------
# Repository State Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo_with_force_push() -> tuple[MockRepository, List[MockCommit]]:
    """Repository fixture demonstrating force push scenario.

    Returns tuple of (repository, old_commits) where old_commits represent
    the commit history before force push.
    """
    repo = small_test_repo_fixture.__wrapped__()  # Get unwrapped fixture

    # Save old commits
    old_commits = repo.commits.copy()

    # Simulate force push - replace commit history
    now = datetime.now(timezone.utc)
    new_commits = [
        MockCommit(
            "force_abc123",
            "Rebased: Initial commit",
            "octocat",
            now - timedelta(days=25),
            ["README.md"],
        ),
        MockCommit(
            "force_def456",
            "Rebased: Add source",
            "octocat",
            now - timedelta(days=20),
            ["src/main.py"],
        ),
    ]

    repo.commits = new_commits

    return repo, old_commits


@pytest.fixture
def repo_with_multiple_branches() -> Dict[str, List[MockCommit]]:
    """Repository with different commit histories per branch.

    Returns dict mapping branch names to commit lists.
    """
    repo = small_test_repo_fixture.__wrapped__()
    now = datetime.now(timezone.utc)

    return {
        "main": repo.commits,
        "develop": repo.commits[:3] + [
            MockCommit(
                "dev_xyz789",
                "Development work in progress",
                "developer",
                now - timedelta(days=1),
                ["src/new_feature.py"],
            )
        ],
        "feature/experimental": [
            repo.commits[0],
            MockCommit(
                "exp_aaa111",
                "Experimental changes",
                "researcher",
                now - timedelta(hours=12),
                ["experiments/test.py"],
            ),
        ],
    }
