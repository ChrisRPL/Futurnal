Summary: Implement dual-mode repository sync: GraphQL API (lightweight) and Git Clone (full fidelity).

# 04 · Repository Sync Strategy

## Purpose
Implement a flexible dual-mode repository synchronization strategy that allows users to choose between lightweight GraphQL API access and full-fidelity git clone operations. Optimize for different use cases: selective file access, metadata-only ingestion, or complete repository analysis.

## Scope
- **GraphQL API Mode**: Selective file fetching, low disk usage, online-only
- **Git Clone Mode**: Complete repository on disk, offline-capable, full history
- Branch and tag selection
- Shallow clone vs full clone options
- Sparse checkout for large repositories
- Incremental updates for both modes
- File content streaming and chunking
- Disk space management and cleanup

## Requirements Alignment
- **Flexible ingestion**: Support both lightweight and full-fidelity sync
- **Offline capability**: Git clone mode enables offline operation
- **Disk efficiency**: Shallow clones and sparse checkout for large repos
- **Rate limit awareness**: GraphQL mode optimizes API usage
- **Privacy**: No repository content in logs

## Sync Modes Comparison

### GraphQL API Mode
**Best for:**
- Selective file access (README, docs, specific paths)
- Metadata-only ingestion (no full content)
- Multiple repository monitoring with limited disk
- Issue/PR tracking without code
- Quick setup and low maintenance

**Characteristics:**
- **Disk usage**: Minimal (only fetched files)
- **API usage**: Medium-high (file content via API)
- **Offline**: No (requires network)
- **History**: Limited to API-accessible data
- **Update speed**: Fast (only changed files)

**Limitations:**
- Cannot access full git history
- Binary files limited by API
- Large files require Git LFS API
- No git operations (diff, blame, etc.)

### Git Clone Mode
**Best for:**
- Complete code analysis
- Full commit history needed
- Offline repository exploration
- Git operations (diff, blame, log)
- Large repositories (one-time clone)

**Characteristics:**
- **Disk usage**: High (complete repository)
- **API usage**: Low (initial metadata only)
- **Offline**: Yes (after initial clone)
- **History**: Complete git history
- **Update speed**: Fast (git fetch)

**Limitations:**
- Higher initial disk usage
- Slower initial setup (clone time)
- Requires git binary
- Submodules add complexity

## Data Model

### SyncStrategy
```python
class SyncStrategy(BaseModel):
    """Configuration for repository sync strategy."""

    # Mode selection
    sync_mode: SyncMode  # GRAPHQL_API or GIT_CLONE

    # Branch selection
    branches: List[str] = Field(default_factory=lambda: ["main"])
    include_all_branches: bool = False
    include_tags: bool = False

    # File selection
    file_patterns: List[str] = Field(
        default_factory=list,
        description="Glob patterns for files to sync (empty = all)"
    )
    exclude_patterns: List[str] = Field(
        default_factory=lambda: [
            ".git/", "node_modules/", "__pycache__/", "*.pyc"
        ]
    )

    # GraphQL API Mode settings
    max_file_size_mb: int = Field(
        default=10,
        description="Skip files larger than this (API mode)"
    )
    fetch_file_content: bool = Field(
        default=True,
        description="Fetch actual file content (vs metadata only)"
    )

    # Git Clone Mode settings
    clone_depth: Optional[int] = Field(
        default=None,
        description="Shallow clone depth (None = full history)"
    )
    use_sparse_checkout: bool = Field(
        default=False,
        description="Use sparse checkout for large repos"
    )
    clone_submodules: bool = Field(
        default=False,
        description="Clone submodules recursively"
    )

class SyncState(BaseModel):
    """Persistent state for repository sync."""

    repo_id: str
    sync_mode: SyncMode
    last_sync_time: datetime
    last_commit_sha: Optional[str] = None  # Latest commit synced
    branch_states: Dict[str, BranchSyncState] = Field(default_factory=dict)

    # Statistics
    total_files_synced: int = 0
    total_bytes_synced: int = 0
    total_commits_processed: int = 0
    sync_errors: int = 0

class BranchSyncState(BaseModel):
    """Sync state for a single branch."""
    branch_name: str
    last_commit_sha: str
    last_sync_time: datetime
    file_count: int = 0
```

## GraphQL API Mode Implementation

### File Tree Fetching
```python
class GraphQLRepositorySync:
    """Sync repository using GitHub GraphQL API."""

    async def sync_repository(
        self,
        descriptor: GitHubRepositoryDescriptor,
        strategy: SyncStrategy,
    ) -> SyncResult:
        """Sync repository using GraphQL API."""
        # Fetch repository tree
        tree = await self._fetch_repository_tree(
            owner=descriptor.owner,
            repo=descriptor.repo,
            ref=descriptor.branches[0],
        )

        # Filter files by patterns
        files_to_sync = self._filter_files(tree, strategy)

        # Fetch file content in batches
        results = []
        for batch in self._batch_files(files_to_sync, batch_size=10):
            contents = await self._fetch_file_contents_batch(batch)
            results.extend(contents)

        return SyncResult(
            files_synced=len(results),
            bytes_synced=sum(len(f.content) for f in results),
            sync_mode="graphql_api",
        )

    async def _fetch_repository_tree(
        self,
        owner: str,
        repo: str,
        ref: str,
    ) -> List[FileEntry]:
        """Fetch repository file tree via GraphQL."""
        query = """
        query($owner: String!, $repo: String!, $expression: String!) {
          repository(owner: $owner, name: $repo) {
            object(expression: $expression) {
              ... on Tree {
                entries {
                  name
                  type
                  mode
                  path
                  object {
                    ... on Blob {
                      byteSize
                      isBinary
                    }
                  }
                }
              }
            }
          }
        }
        """

        variables = {
            "owner": owner,
            "repo": repo,
            "expression": f"{ref}:",
        }

        response = await self.api_client.graphql_request(
            credential_id=descriptor.credential_id,
            query=query,
            variables=variables,
        )

        return self._parse_tree_entries(response)

    async def _fetch_file_contents_batch(
        self,
        files: List[FileEntry],
    ) -> List[FileContent]:
        """Fetch multiple file contents in single GraphQL query."""
        # GraphQL query to fetch multiple files
        # Use aliases to fetch different files in one request
        query = self._build_batch_file_query(files)

        response = await self.api_client.graphql_request(
            credential_id=self.credential_id,
            query=query,
        )

        return self._parse_file_contents(response, files)
```

### GraphQL Query for Batched Files
```graphql
query BatchFileContents($owner: String!, $repo: String!) {
  repository(owner: $owner, name: $repo) {
    file1: object(expression: "main:README.md") {
      ... on Blob {
        text
        byteSize
      }
    }
    file2: object(expression: "main:src/main.py") {
      ... on Blob {
        text
        byteSize
      }
    }
    # ... more files
  }
}
```

## Git Clone Mode Implementation

### Repository Cloning
```python
class GitCloneRepositorySync:
    """Sync repository using git clone."""

    def __init__(
        self,
        *,
        clone_base_dir: Path,
        git_binary: str = "git",
    ):
        self.clone_base_dir = clone_base_dir
        self.git_binary = git_binary

    async def sync_repository(
        self,
        descriptor: GitHubRepositoryDescriptor,
        strategy: SyncStrategy,
    ) -> SyncResult:
        """Sync repository using git clone."""
        repo_dir = self.clone_base_dir / descriptor.id

        if repo_dir.exists():
            # Update existing clone
            return await self._update_repository(repo_dir, descriptor, strategy)
        else:
            # Initial clone
            return await self._clone_repository(repo_dir, descriptor, strategy)

    async def _clone_repository(
        self,
        repo_dir: Path,
        descriptor: GitHubRepositoryDescriptor,
        strategy: SyncStrategy,
    ) -> SyncResult:
        """Perform initial repository clone."""
        clone_url = self._build_clone_url(descriptor)

        # Build git clone command
        cmd = [self.git_binary, "clone"]

        # Shallow clone if configured
        if strategy.clone_depth:
            cmd.extend(["--depth", str(strategy.clone_depth)])

        # Single branch if not cloning all
        if not strategy.include_all_branches and descriptor.branches:
            cmd.extend(["--branch", descriptor.branches[0], "--single-branch"])

        # No tags if not needed
        if not strategy.include_tags:
            cmd.append("--no-tags")

        # Sparse checkout preparation
        if strategy.use_sparse_checkout:
            cmd.append("--filter=blob:none")
            cmd.append("--sparse")

        cmd.extend([clone_url, str(repo_dir)])

        # Execute clone
        start_time = time.time()
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await result.communicate()

        if result.returncode != 0:
            raise GitCloneError(f"Clone failed: {stderr.decode()}")

        # Configure sparse checkout if needed
        if strategy.use_sparse_checkout:
            await self._configure_sparse_checkout(repo_dir, strategy)

        clone_duration = time.time() - start_time

        return SyncResult(
            files_synced=self._count_files(repo_dir),
            bytes_synced=self._calculate_repo_size(repo_dir),
            sync_mode="git_clone",
            duration_seconds=clone_duration,
        )

    async def _update_repository(
        self,
        repo_dir: Path,
        descriptor: GitHubRepositoryDescriptor,
        strategy: SyncStrategy,
    ) -> SyncResult:
        """Update existing cloned repository."""
        # Fetch latest changes
        cmd = [self.git_binary, "-C", str(repo_dir), "fetch", "origin"]

        if strategy.include_tags:
            cmd.append("--tags")

        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await result.communicate()

        # Update working tree for each branch
        for branch in descriptor.branches:
            await self._update_branch(repo_dir, branch)

        return SyncResult(
            files_synced=self._count_files(repo_dir),
            bytes_synced=self._calculate_repo_size(repo_dir),
            sync_mode="git_clone",
        )

    async def _configure_sparse_checkout(
        self,
        repo_dir: Path,
        strategy: SyncStrategy,
    ) -> None:
        """Configure sparse checkout patterns."""
        sparse_checkout_file = repo_dir / ".git" / "info" / "sparse-checkout"

        # Write patterns
        patterns = strategy.file_patterns or ["/*"]
        sparse_checkout_file.write_text("\n".join(patterns))

        # Apply sparse checkout
        cmd = [self.git_binary, "-C", str(repo_dir), "sparse-checkout", "set"]
        cmd.extend(patterns)

        await asyncio.create_subprocess_exec(*cmd)

    def _build_clone_url(
        self,
        descriptor: GitHubRepositoryDescriptor,
    ) -> str:
        """Build authenticated clone URL."""
        # Get credentials
        creds = self.credential_manager.retrieve_credentials(
            descriptor.credential_id
        )

        if isinstance(creds, OAuthTokens):
            token = creds.access_token
        else:
            token = creds.token

        # Build URL with token
        if descriptor.github_host == "github.com":
            return f"https://{token}@github.com/{descriptor.owner}/{descriptor.repo}.git"
        else:
            return f"https://{token}@{descriptor.github_host}/{descriptor.owner}/{descriptor.repo}.git"
```

### Sparse Checkout Patterns
```bash
# Example sparse-checkout patterns
/*                    # Top-level files
docs/                 # Documentation directory
src/                  # Source code
!*.md                 # Exclude markdown files
!tests/               # Exclude tests
```

## Mode Selection Logic

### Automatic Mode Selection
```python
def recommend_sync_mode(
    repo_metadata: RepositoryMetadata,
    available_disk_gb: float,
) -> SyncMode:
    """Recommend sync mode based on repository characteristics."""

    repo_size_gb = repo_metadata.size_kb / (1024 * 1024)

    # Large repository with limited disk space
    if repo_size_gb > available_disk_gb * 0.5:
        return SyncMode.GRAPHQL_API

    # Small repository - clone is efficient
    if repo_size_gb < 0.1:  # < 100 MB
        return SyncMode.GIT_CLONE

    # Many files - GraphQL API may hit rate limits
    if repo_metadata.file_count > 10000:
        return SyncMode.GIT_CLONE

    # Default to GraphQL for most cases
    return SyncMode.GRAPHQL_API
```

## Acceptance Criteria

- ✅ GraphQL API mode fetches files selectively
- ✅ Git clone mode creates complete local repository
- ✅ Shallow clone reduces disk usage for large repos
- ✅ Sparse checkout works for partial repository sync
- ✅ Incremental updates work for both modes
- ✅ Branch selection honored in both modes
- ✅ File pattern filtering works correctly
- ✅ Binary files handled appropriately
- ✅ Git LFS files detected and handled
- ✅ Authenticated clone URLs work with OAuth and PAT
- ✅ Disk space checks prevent out-of-space errors

## Test Plan

### Unit Tests
- Sync mode selection logic
- File pattern matching and filtering
- Clone URL construction with credentials
- Sparse checkout pattern generation
- Disk space calculation

### Integration Tests
- GraphQL API mode with test repository
- Git clone mode with test repository
- Shallow clone with depth limits
- Sparse checkout with patterns
- Branch switching and updates
- Incremental sync (fetch vs re-clone)
- Large file handling (>100MB)
- Binary file detection

### Performance Tests
- Clone time for repositories of various sizes
- GraphQL API batching efficiency
- Incremental update speed (git fetch vs API)
- Disk usage comparison (shallow vs full clone)
- Network bandwidth usage

## Implementation Notes

### GitPython Integration
```python
from git import Repo

# Alternative to subprocess for git operations
repo = Repo.clone_from(
    url=clone_url,
    to_path=repo_dir,
    depth=clone_depth,
    branch=branch_name,
)

# Fetch updates
repo.remotes.origin.fetch()
```

### Disk Space Checks
```python
import shutil

def check_available_disk_space(path: Path) -> float:
    """Get available disk space in GB."""
    stat = shutil.disk_usage(path)
    return stat.free / (1024 ** 3)
```

## Open Questions

- Should we support git submodules recursively?
- How to handle Git LFS in GraphQL API mode?
- Should we implement automatic mode switching based on performance?
- How to handle repository renames/transfers?
- Should we cache GraphQL API responses on disk?
- How to handle monorepos efficiently?

## Dependencies
- GitHubKit for GraphQL API access
- GitPython (`pip install gitpython`) for git operations
- subprocess for git binary invocation (fallback)
- GitHubAPIClientManager for rate-limited requests


