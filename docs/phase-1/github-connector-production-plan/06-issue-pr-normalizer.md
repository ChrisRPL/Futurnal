Summary: Transform GitHub issues and pull requests into PKG entities with semantic relationships.

# 06 · Issue & PR Normalizer

## Purpose
Extract and normalize metadata from GitHub issues and pull requests, transforming them into semantic triples for PKG integration. Capture collaboration dynamics, decision-making processes, and project evolution through structured relationship modeling.

## Scope
- Issue metadata extraction (title, body, labels, milestones, assignees)
- Pull request metadata extraction (reviews, commits, changed files)
- Comment thread reconstruction
- Author and participant entity creation
- Semantic triple generation for PKG
- Timeline event tracking
- Link extraction (mentions, references, external URLs)
- Sentiment analysis (optional)

## Requirements Alignment
- **Collaboration understanding**: Capture participant interactions and decision patterns
- **Project evolution**: Track feature discussions and implementation timelines
- **Knowledge extraction**: Convert unstructured discussions into structured knowledge
- **Privacy**: Redact participant information if configured
- **Performance**: Efficient GraphQL queries to minimize API usage

## Data Model

### IssueMetadata
```python
class IssueState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"

class IssueMetadata(BaseModel):
    """Normalized issue metadata."""

    # Identity
    issue_number: int
    repo_id: str  # Parent repository
    github_url: str

    # Content
    title: str
    body: Optional[str] = None
    body_html: Optional[str] = None  # Rendered HTML

    # Status
    state: IssueState
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime] = None

    # Participants
    author: GitHubUser
    assignees: List[GitHubUser] = Field(default_factory=list)
    participants: List[GitHubUser] = Field(default_factory=list)  # All commenters

    # Classification
    labels: List[Label] = Field(default_factory=list)
    milestone: Optional[Milestone] = None

    # Engagement metrics
    comment_count: int = 0
    reaction_count: int = 0
    reactions: Dict[str, int] = Field(default_factory=dict)  # {"+1": 5, "heart": 2}

    # Links and references
    mentioned_users: List[str] = Field(default_factory=list)
    referenced_issues: List[int] = Field(default_factory=list)
    referenced_prs: List[int] = Field(default_factory=list)
    external_links: List[str] = Field(default_factory=list)

class PullRequestMetadata(BaseModel):
    """Normalized pull request metadata."""

    # Extends IssueMetadata
    pr_number: int
    repo_id: str
    github_url: str

    # Content
    title: str
    body: Optional[str] = None

    # Status
    state: str  # "open", "closed", "merged"
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime] = None
    merged_at: Optional[datetime] = None

    # Participants
    author: GitHubUser
    reviewers: List[GitHubUser] = Field(default_factory=list)
    assignees: List[GitHubUser] = Field(default_factory=list)
    merged_by: Optional[GitHubUser] = None

    # Code changes
    changed_files: int
    additions: int
    deletions: int
    commits: int
    base_branch: str
    head_branch: str

    # Review status
    review_decision: Optional[str] = None  # "APPROVED", "CHANGES_REQUESTED", "REVIEW_REQUIRED"
    approved_by: List[GitHubUser] = Field(default_factory=list)
    changes_requested_by: List[GitHubUser] = Field(default_factory=list)

    # Classification
    labels: List[Label] = Field(default_factory=list)
    milestone: Optional[Milestone] = None

    # Files
    modified_files: List[str] = Field(default_factory=list)

class GitHubUser(BaseModel):
    """Normalized user entity."""
    login: str
    name: Optional[str] = None
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    github_url: str

class Label(BaseModel):
    """Issue/PR label."""
    name: str
    color: str
    description: Optional[str] = None

class Milestone(BaseModel):
    """Project milestone."""
    title: str
    description: Optional[str] = None
    due_on: Optional[datetime] = None
    state: str  # "open", "closed"

class Comment(BaseModel):
    """Issue or PR comment."""
    comment_id: int
    author: GitHubUser
    body: str
    created_at: datetime
    updated_at: datetime
    reactions: Dict[str, int] = Field(default_factory=dict)
```

## Component Design

### IssueNormalizer
```python
class IssueNormalizer:
    """Normalizes GitHub issues into PKG-ready format."""

    def __init__(
        self,
        *,
        api_client: GitHubAPIClientManager,
        extract_comments: bool = True,
        max_comments: int = 100,
    ):
        self.api_client = api_client
        self.extract_comments = extract_comments
        self.max_comments = max_comments

    async def normalize_issue(
        self,
        *,
        repo_owner: str,
        repo_name: str,
        issue_number: int,
        credential_id: str,
    ) -> IssueMetadata:
        """Normalize a single issue."""

        # Fetch issue data via GraphQL
        query = self._build_issue_query()
        variables = {
            "owner": repo_owner,
            "repo": repo_name,
            "number": issue_number,
        }

        response = await self.api_client.graphql_request(
            credential_id=credential_id,
            query=query,
            variables=variables,
        )

        issue_data = response["data"]["repository"]["issue"]

        # Parse and normalize
        metadata = self._parse_issue_data(issue_data, repo_owner, repo_name)

        # Extract comments if requested
        if self.extract_comments:
            metadata.comments = await self._fetch_comments(
                repo_owner, repo_name, issue_number, credential_id
            )

        return metadata

    def _build_issue_query(self) -> str:
        """Build GraphQL query for issue data."""
        return """
        query($owner: String!, $repo: String!, $number: Int!) {
          repository(owner: $owner, name: $repo) {
            issue(number: $number) {
              number
              title
              body
              bodyHTML
              state
              createdAt
              updatedAt
              closedAt
              author {
                login
                ... on User {
                  name
                  email
                  avatarUrl
                }
              }
              assignees(first: 10) {
                nodes {
                  login
                  name
                }
              }
              labels(first: 20) {
                nodes {
                  name
                  color
                  description
                }
              }
              milestone {
                title
                description
                dueOn
                state
              }
              participants(first: 50) {
                nodes {
                  login
                  name
                }
              }
              reactions {
                totalCount
              }
              comments {
                totalCount
              }
            }
          }
        }
        """

    def _parse_issue_data(
        self,
        issue_data: Dict,
        repo_owner: str,
        repo_name: str,
    ) -> IssueMetadata:
        """Parse raw GitHub issue data into normalized format."""

        # Extract author
        author_data = issue_data.get("author", {})
        author = GitHubUser(
            login=author_data.get("login", "ghost"),
            name=author_data.get("name"),
            email=author_data.get("email"),
            avatar_url=author_data.get("avatarUrl"),
            github_url=f"https://github.com/{author_data.get('login', 'ghost')}",
        )

        # Extract labels
        labels = [
            Label(
                name=label["name"],
                color=label["color"],
                description=label.get("description"),
            )
            for label in issue_data.get("labels", {}).get("nodes", [])
        ]

        # Extract milestone
        milestone_data = issue_data.get("milestone")
        milestone = None
        if milestone_data:
            milestone = Milestone(
                title=milestone_data["title"],
                description=milestone_data.get("description"),
                due_on=milestone_data.get("dueOn"),
                state=milestone_data["state"],
            )

        return IssueMetadata(
            issue_number=issue_data["number"],
            repo_id=f"{repo_owner}/{repo_name}",
            github_url=f"https://github.com/{repo_owner}/{repo_name}/issues/{issue_data['number']}",
            title=issue_data["title"],
            body=issue_data.get("body"),
            body_html=issue_data.get("bodyHTML"),
            state=IssueState(issue_data["state"].lower()),
            created_at=datetime.fromisoformat(issue_data["createdAt"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(issue_data["updatedAt"].replace("Z", "+00:00")),
            closed_at=datetime.fromisoformat(issue_data["closedAt"].replace("Z", "+00:00"))
                if issue_data.get("closedAt") else None,
            author=author,
            labels=labels,
            milestone=milestone,
            comment_count=issue_data.get("comments", {}).get("totalCount", 0),
            reaction_count=issue_data.get("reactions", {}).get("totalCount", 0),
        )
```

### PullRequestNormalizer
```python
class PullRequestNormalizer:
    """Normalizes GitHub pull requests into PKG-ready format."""

    async def normalize_pull_request(
        self,
        *,
        repo_owner: str,
        repo_name: str,
        pr_number: int,
        credential_id: str,
    ) -> PullRequestMetadata:
        """Normalize a single pull request."""

        query = self._build_pr_query()
        variables = {
            "owner": repo_owner,
            "repo": repo_name,
            "number": pr_number,
        }

        response = await self.api_client.graphql_request(
            credential_id=credential_id,
            query=query,
            variables=variables,
        )

        pr_data = response["data"]["repository"]["pullRequest"]
        return self._parse_pr_data(pr_data, repo_owner, repo_name)

    def _build_pr_query(self) -> str:
        """Build GraphQL query for PR data."""
        return """
        query($owner: String!, $repo: String!, $number: Int!) {
          repository(owner: $owner, name: $repo) {
            pullRequest(number: $number) {
              number
              title
              body
              state
              createdAt
              updatedAt
              closedAt
              mergedAt
              author {
                login
                ... on User {
                  name
                }
              }
              reviewDecision
              reviews(first: 20) {
                nodes {
                  author {
                    login
                  }
                  state
                }
              }
              changedFiles
              additions
              deletions
              commits {
                totalCount
              }
              baseRefName
              headRefName
              mergedBy {
                login
                name
              }
              labels(first: 20) {
                nodes {
                  name
                  color
                }
              }
              files(first: 100) {
                nodes {
                  path
                }
              }
            }
          }
        }
        """
```

## Semantic Triple Generation

### Triple Extraction
```python
def extract_issue_triples(issue: IssueMetadata) -> List[Triple]:
    """Generate semantic triples from issue metadata."""

    triples = []

    # Issue entity
    issue_uri = f"github:issue:{issue.repo_id}:{issue.issue_number}"

    # Basic properties
    triples.append(Triple(
        subject=issue_uri,
        predicate="rdf:type",
        object="github:Issue",
    ))
    triples.append(Triple(
        subject=issue_uri,
        predicate="dc:title",
        object=issue.title,
    ))

    # Author relationship
    author_uri = f"github:user:{issue.author.login}"
    triples.append(Triple(
        subject=issue_uri,
        predicate="github:createdBy",
        object=author_uri,
    ))
    triples.append(Triple(
        subject=author_uri,
        predicate="rdf:type",
        object="github:User",
    ))

    # Labels (classification)
    for label in issue.labels:
        label_uri = f"github:label:{issue.repo_id}:{label.name}"
        triples.append(Triple(
            subject=issue_uri,
            predicate="github:hasLabel",
            object=label_uri,
        ))

    # Milestone (project phase)
    if issue.milestone:
        milestone_uri = f"github:milestone:{issue.repo_id}:{issue.milestone.title}"
        triples.append(Triple(
            subject=issue_uri,
            predicate="github:inMilestone",
            object=milestone_uri,
        ))

    # Temporal properties
    triples.append(Triple(
        subject=issue_uri,
        predicate="dc:created",
        object=issue.created_at.isoformat(),
    ))

    return triples
```

## Acceptance Criteria

- ✅ Issues extracted with complete metadata
- ✅ Pull requests extracted with review status
- ✅ Comment threads reconstructed with authors
- ✅ Labels and milestones normalized
- ✅ Participant relationships captured
- ✅ Semantic triples generated for PKG
- ✅ Timeline events ordered chronologically
- ✅ References (mentions, issue links) extracted
- ✅ GraphQL queries optimized for rate limits
- ✅ Privacy settings honored (author anonymization)

## Test Plan

### Unit Tests
- GraphQL query construction
- Issue metadata parsing
- PR metadata parsing
- Triple generation from metadata
- User entity deduplication

### Integration Tests
- Fetch real issues from test repository
- Fetch real PRs with reviews
- Comment thread extraction
- Label and milestone handling
- Reference extraction (mentions, issue links)

## Implementation Notes

### GraphQL Optimization
- Fetch issues in batches (100 at a time)
- Use cursor-based pagination
- Include rate limit info in queries

### Link Extraction
```python
def extract_issue_references(text: str) -> List[int]:
    """Extract issue/PR numbers from text."""
    pattern = r'#(\d+)'
    return [int(m) for m in re.findall(pattern, text)]
```

## Open Questions

- Should we analyze sentiment of comments?
- How deep to fetch comment threads (nested replies)?
- Should we track issue/PR state transitions?
- How to handle cross-repository references?

## Dependencies
- GitHubAPIClientManager for GraphQL queries
- Triple extraction utilities
- (Optional) Sentiment analysis library


