Summary: Implement incremental sync with commit SHA tracking, delta detection, and force-push handling.

# 07 · Incremental Sync Engine

## Purpose
Implement efficient incremental synchronization using commit SHA tracking to process only new or changed content. Detect force pushes, branch deletions, and handle conflict resolution while maintaining sync state integrity.

## Scope
- Commit SHA-based state tracking per branch
- Delta sync (only process new commits)
- Force-push and rebase detection
- Branch divergence handling
- Deleted branch cleanup
- File-level change detection
- Sync state persistence
- 5-minute detection window for updates

## Requirements Alignment
- **5-minute detection**: New commits detected within 5 minutes
- **Incremental learning**: Only process changed content
- **Resilience**: Handle force-pushes and rebases gracefully
- **Offline capable**: Queue sync operations for later
- **Privacy**: State stored locally with audit trail

## Data Model

### RepositorySyncState
```python
class RepositorySyncState(BaseModel):
    """Persistent sync state for repository."""

    # Identity
    repo_id: str
    repo_full_name: str  # owner/repo

    # Sync configuration
    sync_mode: SyncMode
    last_full_sync: Optional[datetime] = None

    # Branch states
    branch_states: Dict[str, BranchSyncState] = Field(default_factory=dict)

    # Global statistics
    total_commits_processed: int = 0
    total_files_synced: int = 0
    total_sync_operations: int = 0
    last_sync_time: datetime

class BranchSyncState(BaseModel):
    """Sync state for a single branch."""

    branch_name: str
    last_commit_sha: str
    last_sync_time: datetime

    # Commit tracking
    commits_processed: int = 0
    files_modified: int = 0

    # Divergence detection
    parent_commit_sha: Optional[str] = None  # Previous HEAD before update
    force_push_detected: bool = False

class SyncResult(BaseModel):
    """Result of sync operation."""

    repo_id: str
    branch: str
    sync_mode: str

    # Changes detected
    new_commits: List[str] = Field(default_factory=list)  # Commit SHAs
    modified_files: List[str] = Field(default_factory=list)
    deleted_files: List[str] = Field(default_factory=list)
    added_files: List[str] = Field(default_factory=list)

    # Sync statistics
    commits_processed: int = 0
    files_synced: int = 0
    bytes_synced: int = 0
    sync_duration_seconds: float = 0.0

    # Sync status
    force_push_handled: bool = False
    errors: List[str] = Field(default_factory=list)
```

## Component Design

### IncrementalSyncEngine
```python
class IncrementalSyncEngine:
    """Manages incremental repository synchronization."""

    def __init__(
        self,
        *,
        api_client: GitHubAPIClientManager,
        state_store: StateStore,
        file_classifier: FileClassifier,
        element_sink: ElementSink,
    ):
        self.api_client = api_client
        self.state_store = state_store
        self.file_classifier = file_classifier
        self.element_sink = element_sink

    async def sync_repository(
        self,
        descriptor: GitHubRepositoryDescriptor,
    ) -> SyncResult:
        """Perform incremental sync of repository."""

        # Load sync state
        state = self._load_sync_state(descriptor.id)

        # Sync each configured branch
        results = []
        for branch in descriptor.branches:
            result = await self._sync_branch(
                descriptor, branch, state
            )
            results.append(result)

        # Save updated state
        self._save_sync_state(state)

        return self._merge_results(results)

    async def _sync_branch(
        self,
        descriptor: GitHubRepositoryDescriptor,
        branch_name: str,
        state: RepositorySyncState,
    ) -> SyncResult:
        """Sync a single branch incrementally."""

        start_time = time.time()

        # Get current branch HEAD
        current_head = await self._get_branch_head(
            descriptor.owner,
            descriptor.repo,
            branch_name,
            descriptor.credential_id,
        )

        # Check if branch exists in state
        branch_state = state.branch_states.get(branch_name)

        if not branch_state:
            # First sync of this branch
            result = await self._initial_branch_sync(
                descriptor, branch_name, current_head
            )
        elif branch_state.last_commit_sha == current_head:
            # No changes
            result = SyncResult(
                repo_id=descriptor.id,
                branch=branch_name,
                sync_mode=descriptor.sync_mode.value,
            )
        else:
            # Incremental sync
            result = await self._incremental_branch_sync(
                descriptor, branch_name, branch_state, current_head
            )

        result.sync_duration_seconds = time.time() - start_time

        # Update branch state
        state.branch_states[branch_name] = BranchSyncState(
            branch_name=branch_name,
            last_commit_sha=current_head,
            last_sync_time=datetime.utcnow(),
            commits_processed=branch_state.commits_processed + result.commits_processed
                if branch_state else result.commits_processed,
            files_modified=len(result.modified_files),
        )

        return result

    async def _get_branch_head(
        self,
        owner: str,
        repo: str,
        branch: str,
        credential_id: str,
    ) -> str:
        """Get current HEAD commit SHA for branch."""

        query = """
        query($owner: String!, $repo: String!, $branch: String!) {
          repository(owner: $owner, name: $repo) {
            ref(qualifiedName: $branch) {
              target {
                ... on Commit {
                  oid
                }
              }
            }
          }
        }
        """

        variables = {
            "owner": owner,
            "repo": repo,
            "branch": f"refs/heads/{branch}",
        }

        response = await self.api_client.graphql_request(
            credential_id=credential_id,
            query=query,
            variables=variables,
        )

        return response["data"]["repository"]["ref"]["target"]["oid"]

    async def _incremental_branch_sync(
        self,
        descriptor: GitHubRepositoryDescriptor,
        branch_name: str,
        branch_state: BranchSyncState,
        current_head: str,
    ) -> SyncResult:
        """Sync only new commits since last sync."""

        # Get commit history from last_commit_sha to current_head
        commits = await self._get_commit_range(
            descriptor.owner,
            descriptor.repo,
            branch_state.last_commit_sha,
            current_head,
            descriptor.credential_id,
        )

        # Check for force push (commits not in history)
        if self._is_force_push(commits, branch_state.last_commit_sha):
            logger.warning(
                f"Force push detected on {branch_name}, "
                f"performing full resync"
            )
            return await self._handle_force_push(
                descriptor, branch_name, current_head
            )

        # Process each new commit
        modified_files = set()
        deleted_files = set()
        added_files = set()

        for commit in commits:
            changes = await self._get_commit_changes(
                descriptor.owner,
                descriptor.repo,
                commit["oid"],
                descriptor.credential_id,
            )

            modified_files.update(changes["modified"])
            deleted_files.update(changes["deleted"])
            added_files.update(changes["added"])

        # Sync changed files
        await self._sync_files(
            descriptor,
            list(modified_files | added_files),
        )

        # Handle deletions
        await self._handle_deleted_files(
            descriptor,
            list(deleted_files),
        )

        return SyncResult(
            repo_id=descriptor.id,
            branch=branch_name,
            sync_mode=descriptor.sync_mode.value,
            new_commits=[c["oid"] for c in commits],
            modified_files=list(modified_files),
            deleted_files=list(deleted_files),
            added_files=list(added_files),
            commits_processed=len(commits),
            files_synced=len(modified_files) + len(added_files),
        )

    async def _get_commit_range(
        self,
        owner: str,
        repo: str,
        from_sha: str,
        to_sha: str,
        credential_id: str,
    ) -> List[Dict]:
        """Get commits between two SHAs."""

        query = """
        query($owner: String!, $repo: String!, $since: GitTimestamp!) {
          repository(owner: $owner, name: $repo) {
            ref(qualifiedName: "refs/heads/main") {
              target {
                ... on Commit {
                  history(first: 100, since: $since) {
                    nodes {
                      oid
                      message
                      committedDate
                      author {
                        name
                        email
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """

        # Note: GraphQL doesn't support commit range queries directly
        # Alternative: Use REST API or fetch all commits and filter

        response = await self.api_client.rest_request(
            credential_id=credential_id,
            method="GET",
            endpoint=f"/repos/{owner}/{repo}/compare/{from_sha}...{to_sha}",
        )

        return response.get("commits", [])

    async def _get_commit_changes(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
        credential_id: str,
    ) -> Dict[str, List[str]]:
        """Get file changes in a commit."""

        response = await self.api_client.rest_request(
            credential_id=credential_id,
            method="GET",
            endpoint=f"/repos/{owner}/{repo}/commits/{commit_sha}",
        )

        changes = {
            "added": [],
            "modified": [],
            "deleted": [],
        }

        for file in response.get("files", []):
            status = file["status"]
            if status == "added":
                changes["added"].append(file["filename"])
            elif status == "modified":
                changes["modified"].append(file["filename"])
            elif status == "removed":
                changes["deleted"].append(file["filename"])

        return changes

    def _is_force_push(
        self,
        commits: List[Dict],
        expected_parent_sha: str,
    ) -> bool:
        """Detect if force push occurred."""

        # If expected parent SHA not in commit history, it's a force push
        commit_shas = {c["oid"] for c in commits}
        return expected_parent_sha not in commit_shas

    async def _handle_force_push(
        self,
        descriptor: GitHubRepositoryDescriptor,
        branch_name: str,
        current_head: str,
    ) -> SyncResult:
        """Handle force push by performing full resync."""

        # Full resync of branch
        result = await self._initial_branch_sync(
            descriptor, branch_name, current_head
        )
        result.force_push_handled = True

        return result

    async def _initial_branch_sync(
        self,
        descriptor: GitHubRepositoryDescriptor,
        branch_name: str,
        head_sha: str,
    ) -> SyncResult:
        """Perform initial sync of branch."""

        # Depending on sync mode, use different strategies
        if descriptor.sync_mode == SyncMode.GRAPHQL_API:
            return await self._graphql_full_sync(descriptor, branch_name)
        else:
            return await self._git_clone_full_sync(descriptor, branch_name)
```

## Force Push Detection

### Detection Algorithm
```python
def detect_force_push(
    previous_head: str,
    new_head: str,
    commit_history: List[str],
) -> bool:
    """Detect if a force push occurred."""

    # If previous HEAD not in commit history leading to new HEAD,
    # then the branch was force-pushed

    return previous_head not in commit_history
```

### Force Push Handling Strategy
1. **Detect**: Check if previous HEAD is in commit history
2. **Alert**: Log force push event with audit trail
3. **Resync**: Perform full branch resync
4. **Reconcile**: Mark old commits as stale in PKG
5. **Update**: Reset branch state to new HEAD

## Acceptance Criteria

- ✅ Incremental sync processes only new commits
- ✅ Force push detected and handled correctly
- ✅ Branch deletions detected and cleaned up
- ✅ File-level changes (added/modified/deleted) tracked
- ✅ Sync state persisted reliably
- ✅ 5-minute detection window for new commits
- ✅ Concurrent branch syncs handled safely
- ✅ Rebase operations detected as force pushes
- ✅ Merge commits processed correctly
- ✅ Sync state survives process restarts

## Test Plan

### Unit Tests
- Commit SHA comparison
- Force push detection logic
- File change categorization
- State persistence and loading
- Commit range calculation

### Integration Tests
- Incremental sync with new commits
- Force push handling and resync
- Branch deletion cleanup
- Multi-branch concurrent sync
- Large commit history (>1000 commits)
- State recovery after failure

### Stress Tests
- High-frequency commits (>100/hour)
- Large file changes (>10MB diffs)
- Many branches (>50) syncing concurrently

## Implementation Notes

### State Storage
```python
# Store state in JSON file per repository
state_path = workspace / "sync_state" / f"{repo_id}.json"
```

### Commit History Caching
```python
# Cache commit history to avoid re-fetching
commit_cache: Dict[str, List[str]] = {}
```

## Open Questions

- Should we support cherry-pick detection?
- How to handle octopus merges (>2 parents)?
- Should we track which commits introduced which files?
- How to optimize large repository sync (>100k commits)?
- Should we support partial branch sync (only recent commits)?

## Dependencies
- GitHubAPIClientManager for API access
- StateStore for persistence
- FileClassifier for file type detection
- ElementSink for processed content


