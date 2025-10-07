Summary: Implement GitHub webhook integration for real-time repository updates (optional).

# 08 · Webhook Integration

## Purpose
Implement optional GitHub webhook support for real-time repository updates, eliminating polling delays and reducing API usage. Webhooks provide immediate notifications for pushes, PRs, issues, and releases.

## Scope
- Local webhook receiver server (configurable port)
- GitHub webhook signature verification (HMAC-SHA256)
- Event filtering and routing
- Webhook payload parsing
- Event queue for async processing
- Fallback to polling if webhooks unavailable
- ngrok/tunneling support for development
- Security and DDoS protection

## Requirements Alignment
- **Real-time updates**: Detect changes within seconds (vs 5-minute polling)
- **API efficiency**: Webhooks eliminate polling, save rate limit
- **Optional feature**: Works with or without webhooks
- **Security**: HMAC signature verification, secret management
- **Privacy**: Webhook payloads logged without sensitive data

## Webhook Events

### Supported Events
```python
class WebhookEventType(str, Enum):
    PUSH = "push"                       # New commits
    PULL_REQUEST = "pull_request"       # PR opened/updated/closed
    ISSUES = "issues"                   # Issue opened/updated/closed
    ISSUE_COMMENT = "issue_comment"     # Comment on issue/PR
    RELEASE = "release"                 # Release published
    CREATE = "create"                   # Branch/tag created
    DELETE = "delete"                   # Branch/tag deleted
    REPOSITORY = "repository"           # Repo settings changed
```

## Data Model

### WebhookConfig
```python
class WebhookConfig(BaseModel):
    """Configuration for webhook integration."""

    # Server settings
    enabled: bool = False
    listen_host: str = "127.0.0.1"  # Localhost only by default
    listen_port: int = 8765
    public_url: Optional[str] = None  # For GitHub webhook configuration

    # Security
    secret: str  # HMAC secret for signature verification
    verify_signature: bool = True

    # Event filtering
    enabled_events: List[WebhookEventType] = Field(
        default_factory=lambda: [
            WebhookEventType.PUSH,
            WebhookEventType.PULL_REQUEST,
            WebhookEventType.ISSUES,
        ]
    )

    # Processing
    queue_size: int = 1000
    process_async: bool = True

class WebhookEvent(BaseModel):
    """Parsed webhook event."""

    event_id: str
    event_type: WebhookEventType
    repository: str  # owner/repo
    timestamp: datetime

    # Event-specific payload
    payload: Dict[str, Any]

    # Processing status
    processed: bool = False
    processing_error: Optional[str] = None
```

## Component Design

### WebhookServer
```python
from aiohttp import web
import hmac
import hashlib

class WebhookServer:
    """Receives and processes GitHub webhooks."""

    def __init__(
        self,
        config: WebhookConfig,
        event_handler: WebhookEventHandler,
    ):
        self.config = config
        self.event_handler = event_handler
        self.event_queue: asyncio.Queue = asyncio.Queue(
            maxsize=config.queue_size
        )
        self.app = web.Application()
        self._setup_routes()

    def _setup_routes(self):
        """Setup HTTP routes."""
        self.app.router.add_post("/webhook/github", self.handle_webhook)
        self.app.router.add_get("/health", self.health_check)

    async def handle_webhook(self, request: web.Request) -> web.Response:
        """Handle incoming GitHub webhook."""

        # Verify signature
        if self.config.verify_signature:
            if not self._verify_signature(request):
                logger.warning("Invalid webhook signature")
                return web.Response(status=401, text="Invalid signature")

        # Parse event
        event_type = request.headers.get("X-GitHub-Event")
        delivery_id = request.headers.get("X-GitHub-Delivery")
        payload = await request.json()

        # Filter events
        if event_type not in [e.value for e in self.config.enabled_events]:
            return web.Response(status=200, text="Event ignored")

        # Create event
        event = WebhookEvent(
            event_id=delivery_id,
            event_type=WebhookEventType(event_type),
            repository=payload["repository"]["full_name"],
            timestamp=datetime.utcnow(),
            payload=payload,
        )

        # Queue for processing
        try:
            await asyncio.wait_for(
                self.event_queue.put(event),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            logger.error("Event queue full, dropping event")
            return web.Response(status=503, text="Queue full")

        return web.Response(status=200, text="Event queued")

    def _verify_signature(self, request: web.Request) -> bool:
        """Verify GitHub webhook signature."""

        signature_header = request.headers.get("X-Hub-Signature-256")
        if not signature_header:
            return False

        # Extract signature
        try:
            _, signature = signature_header.split("=")
        except ValueError:
            return False

        # Compute expected signature
        body = await request.read()
        expected = hmac.new(
            self.config.secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()

        # Constant-time comparison
        return hmac.compare_digest(signature, expected)

    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "queue_size": self.event_queue.qsize(),
            "max_queue_size": self.config.queue_size,
        })

    async def start(self):
        """Start webhook server."""
        runner = web.AppRunner(self.app)
        await runner.setup()

        site = web.TCPSite(
            runner,
            self.config.listen_host,
            self.config.listen_port,
        )
        await site.start()

        logger.info(
            f"Webhook server listening on "
            f"{self.config.listen_host}:{self.config.listen_port}"
        )

        # Start event processor
        asyncio.create_task(self._process_events())

    async def _process_events(self):
        """Process queued webhook events."""
        while True:
            event = await self.event_queue.get()

            try:
                await self.event_handler.handle_event(event)
                event.processed = True
            except Exception as e:
                logger.error(f"Error processing webhook event: {e}")
                event.processing_error = str(e)
            finally:
                self.event_queue.task_done()
```

### WebhookEventHandler
```python
class WebhookEventHandler:
    """Routes webhook events to appropriate handlers."""

    def __init__(
        self,
        sync_engine: IncrementalSyncEngine,
        issue_normalizer: IssueNormalizer,
        pr_normalizer: PullRequestNormalizer,
    ):
        self.sync_engine = sync_engine
        self.issue_normalizer = issue_normalizer
        self.pr_normalizer = pr_normalizer

    async def handle_event(self, event: WebhookEvent):
        """Route event to appropriate handler."""

        if event.event_type == WebhookEventType.PUSH:
            await self._handle_push(event)
        elif event.event_type == WebhookEventType.PULL_REQUEST:
            await self._handle_pull_request(event)
        elif event.event_type == WebhookEventType.ISSUES:
            await self._handle_issue(event)
        elif event.event_type == WebhookEventType.CREATE:
            await self._handle_create(event)
        elif event.event_type == WebhookEventType.DELETE:
            await self._handle_delete(event)
        else:
            logger.debug(f"Unhandled event type: {event.event_type}")

    async def _handle_push(self, event: WebhookEvent):
        """Handle push event (new commits)."""

        payload = event.payload
        repo_full_name = payload["repository"]["full_name"]
        ref = payload["ref"]  # e.g., "refs/heads/main"
        branch = ref.replace("refs/heads/", "")

        # Trigger incremental sync for this branch
        descriptor = self._get_repository_descriptor(repo_full_name)
        if descriptor and branch in descriptor.branches:
            await self.sync_engine._sync_branch(
                descriptor, branch, load_sync_state(descriptor.id)
            )

    async def _handle_pull_request(self, event: WebhookEvent):
        """Handle pull request event."""

        payload = event.payload
        action = payload["action"]  # opened, closed, synchronize, etc.
        pr_number = payload["pull_request"]["number"]
        repo = payload["repository"]["full_name"]

        # Normalize and update PR
        owner, repo_name = repo.split("/")
        descriptor = self._get_repository_descriptor(repo)

        if descriptor:
            pr_metadata = await self.pr_normalizer.normalize_pull_request(
                repo_owner=owner,
                repo_name=repo_name,
                pr_number=pr_number,
                credential_id=descriptor.credential_id,
            )

            # Send to PKG
            # ... (implementation)

    async def _handle_issue(self, event: WebhookEvent):
        """Handle issue event."""
        # Similar to pull request handling
        pass

    async def _handle_create(self, event: WebhookEvent):
        """Handle branch/tag creation."""
        payload = event.payload
        ref_type = payload["ref_type"]  # "branch" or "tag"
        ref = payload["ref"]

        if ref_type == "branch":
            # Check if we should sync this new branch
            pass

    async def _handle_delete(self, event: WebhookEvent):
        """Handle branch/tag deletion."""
        payload = event.payload
        ref_type = payload["ref_type"]
        ref = payload["ref"]

        if ref_type == "branch":
            # Clean up branch state and data
            pass
```

## GitHub Webhook Configuration

### Setup via GitHub API
```python
async def configure_webhook(
    owner: str,
    repo: str,
    webhook_url: str,
    secret: str,
    credential_id: str,
) -> Dict:
    """Configure webhook via GitHub API."""

    payload = {
        "config": {
            "url": webhook_url,
            "content_type": "json",
            "secret": secret,
            "insecure_ssl": "0",  # Require HTTPS
        },
        "events": [
            "push",
            "pull_request",
            "issues",
            "issue_comment",
            "release",
            "create",
            "delete",
        ],
        "active": True,
    }

    response = await api_client.rest_request(
        credential_id=credential_id,
        method="POST",
        endpoint=f"/repos/{owner}/{repo}/hooks",
        data=payload,
    )

    return response
```

### CLI Setup Command
```bash
$ futurnal sources github webhook enable <repo_id> \
  --port 8765 \
  --public-url https://my-domain.com/webhook/github

✓ Webhook server configured
✓ GitHub webhook created
✓ Real-time updates enabled

Webhook URL: https://my-domain.com/webhook/github
Secret: xxxxxxxxxx (stored in keychain)
```

## Development Setup (ngrok)

### ngrok Tunnel
```bash
# Start ngrok tunnel
ngrok http 8765

# Get public URL
https://abc123.ngrok.io -> http://localhost:8765

# Configure webhook with ngrok URL
futurnal sources github webhook enable <repo_id> \
  --public-url https://abc123.ngrok.io/webhook/github
```

## Security Considerations

### Signature Verification
```python
def verify_webhook_signature(
    payload_body: bytes,
    signature_header: str,
    secret: str,
) -> bool:
    """Verify GitHub webhook signature."""

    # Extract signature
    if not signature_header.startswith("sha256="):
        return False

    signature = signature_header[7:]  # Remove "sha256=" prefix

    # Compute expected signature
    expected = hmac.new(
        secret.encode(),
        payload_body,
        hashlib.sha256,
    ).hexdigest()

    # Constant-time comparison to prevent timing attacks
    return hmac.compare_digest(signature, expected)
```

### Rate Limiting
```python
class WebhookRateLimiter:
    """Rate limit webhook requests per repository."""

    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.request_times: Dict[str, deque] = {}

    def allow_request(self, repo_full_name: str) -> bool:
        """Check if request should be allowed."""
        now = time.time()

        if repo_full_name not in self.request_times:
            self.request_times[repo_full_name] = deque()

        times = self.request_times[repo_full_name]

        # Remove old requests (>1 minute)
        while times and times[0] < now - 60:
            times.popleft()

        # Check limit
        if len(times) >= self.max_requests:
            return False

        times.append(now)
        return True
```

## Acceptance Criteria

- ✅ Webhook server starts and listens on configured port
- ✅ HMAC-SHA256 signature verification works
- ✅ Push events trigger incremental sync
- ✅ PR events update PR metadata
- ✅ Issue events update issue metadata
- ✅ Event queue handles bursts (>100 events/minute)
- ✅ Webhook configuration via GitHub API works
- ✅ Fallback to polling if webhooks disabled
- ✅ Health check endpoint responds
- ✅ Rate limiting prevents abuse

## Test Plan

### Unit Tests
- Signature verification (valid/invalid)
- Event parsing and routing
- Queue management (full queue handling)
- Rate limiting logic

### Integration Tests
- End-to-end webhook flow with test events
- GitHub webhook configuration API
- Event handler routing
- Async processing
- Health check endpoint

### Security Tests
- Invalid signature rejection
- Replay attack prevention
- Rate limiting enforcement
- DDoS protection

### Manual Tests
- ngrok tunnel setup
- GitHub webhook delivery
- Real-time push detection
- PR and issue updates

## Implementation Notes

### Webhook vs Polling Comparison
- **Latency**: Webhooks < 1s, Polling 5 minutes
- **API usage**: Webhooks ~0 requests, Polling 12/hour
- **Reliability**: Webhooks depend on network, Polling always works

### Recommended Setup
- Development: ngrok tunnels
- Production: Public server with HTTPS
- Fallback: Always keep polling enabled

## Open Questions

- Should we implement webhook retries for failed processing?
- How to handle webhook flooding (DDoS)?
- Should we support webhook event replay?
- How to migrate from polling to webhooks seamlessly?

## Dependencies
- aiohttp (`pip install aiohttp`) for webhook server
- hmac (stdlib) for signature verification
- GitHubAPIClientManager for webhook configuration
- IncrementalSyncEngine for triggered syncs


