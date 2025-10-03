Summary: Reconstruct email conversation threads using Message-ID graph for Ghost's conversational understanding.

# 05 · Thread Reconstruction Engine

## Purpose
Reconstruct email conversation threads from Message-ID, References, and In-Reply-To headers per RFC 2822, building a conversation graph that enables the Ghost to understand communication patterns, relationship dynamics, and conversational context evolution over time.

## Scope
- Message-ID graph construction
- References/In-Reply-To header parsing and validation
- Thread assembly algorithm (tree construction)
- Participant extraction and role identification
- Subject evolution tracking
- Temporal conversation analysis
- Thread metadata extraction for semantic triples
- PKG integration for conversation relationships

## Requirements Alignment
- **Conversational context**: Enable Ghost to understand email threads as cohesive conversations
- **Relationship dynamics**: Extract participant patterns and communication flows
- **Temporal awareness**: Track conversation evolution over time
- **Privacy-first**: Thread analysis without exposing email bodies
- **PKG integration**: Store threads as graph relationships

## Data Model

### EmailThread
```python
class EmailThread(BaseModel):
    """Represents a reconstructed email conversation thread."""

    # Identity
    thread_id: str  # Root Message-ID or generated ID
    root_message_id: str  # First message in thread

    # Messages
    message_ids: List[str] = Field(default_factory=list)
    message_count: int = 0

    # Participants
    participants: List[ThreadParticipant] = Field(default_factory=list)
    participant_count: int = 0

    # Thread metadata
    subject: str  # Original subject (without Re:/Fwd:)
    subject_variations: List[str] = Field(default_factory=list)
    start_date: datetime
    last_message_date: datetime
    duration_days: float = 0.0

    # Thread structure
    depth: int = 0  # Maximum depth of thread tree
    branch_count: int = 0  # Number of branches

    # Analysis
    total_response_time_minutes: float = 0.0
    average_response_time_minutes: float = 0.0
    has_attachments: bool = False

    # Provenance
    mailbox_id: str
    reconstructed_at: datetime


class ThreadParticipant(BaseModel):
    """Participant in an email thread."""
    email_address: str
    display_name: Optional[str] = None
    role: ParticipantRole
    message_count: int = 0
    first_message_date: datetime
    last_message_date: datetime


class ParticipantRole(str, Enum):
    INITIATOR = "initiator"  # Started the thread
    PRIMARY_RECIPIENT = "primary_recipient"  # In To: of root message
    PARTICIPANT = "participant"  # Active participant (sent messages)
    CC_RECIPIENT = "cc_recipient"  # Only in Cc:
    OBSERVER = "observer"  # Only in Bcc: (if known)


class ThreadNode(BaseModel):
    """Node in thread tree structure."""
    message_id: str
    parent_message_id: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    depth: int = 0
    date: datetime
    from_address: str
    subject: str
```

## Component Design

### ThreadReconstructor
```python
class ThreadReconstructor:
    """Reconstruct conversation threads from email messages."""

    def __init__(self):
        self.message_graph: Dict[str, ThreadNode] = {}
        self.threads: Dict[str, EmailThread] = {}

    def add_message(self, email_message: EmailMessage) -> None:
        """Add message to thread graph."""
        node = ThreadNode(
            message_id=email_message.message_id,
            parent_message_id=email_message.in_reply_to,
            date=email_message.date,
            from_address=email_message.from_address.address,
            subject=email_message.subject or "",
        )

        # Add to graph
        self.message_graph[email_message.message_id] = node

        # Link to parent
        if email_message.in_reply_to and email_message.in_reply_to in self.message_graph:
            parent = self.message_graph[email_message.in_reply_to]
            parent.children.append(email_message.message_id)

        # Try to link via References header if In-Reply-To missing
        elif email_message.references:
            self._link_via_references(email_message)

    def _link_via_references(self, email_message: EmailMessage) -> None:
        """Link message to parent via References header."""
        # References are ordered from oldest to newest
        # Last reference is the immediate parent
        for ref in reversed(email_message.references):
            if ref in self.message_graph:
                node = self.message_graph[email_message.message_id]
                node.parent_message_id = ref
                parent = self.message_graph[ref]
                parent.children.append(email_message.message_id)
                break

    def reconstruct_threads(self) -> List[EmailThread]:
        """Reconstruct all threads from message graph."""
        # Find root messages (no parent)
        root_messages = [
            node for node in self.message_graph.values()
            if not node.parent_message_id
        ]

        threads = []
        for root in root_messages:
            thread = self._build_thread(root)
            threads.append(thread)

        self.threads = {t.thread_id: t for t in threads}
        return threads

    def _build_thread(self, root: ThreadNode) -> EmailThread:
        """Build thread from root message."""
        # Collect all messages in thread
        message_ids = []
        participants_map: Dict[str, ThreadParticipant] = {}

        def traverse(node: ThreadNode, depth: int = 0):
            message_ids.append(node.message_id)

            # Track participant
            if node.from_address not in participants_map:
                participants_map[node.from_address] = ThreadParticipant(
                    email_address=node.from_address,
                    role=ParticipantRole.INITIATOR if depth == 0 else ParticipantRole.PARTICIPANT,
                    message_count=0,
                    first_message_date=node.date,
                    last_message_date=node.date,
                )

            participant = participants_map[node.from_address]
            participant.message_count += 1
            participant.last_message_date = max(participant.last_message_date, node.date)

            # Traverse children
            for child_id in node.children:
                if child_id in self.message_graph:
                    traverse(self.message_graph[child_id], depth + 1)

        traverse(root)

        # Calculate thread metadata
        dates = [self.message_graph[mid].date for mid in message_ids if mid in self.message_graph]
        start_date = min(dates)
        last_date = max(dates)
        duration_days = (last_date - start_date).total_seconds() / 86400

        # Calculate tree depth
        def calc_depth(node: ThreadNode) -> int:
            if not node.children:
                return 0
            return 1 + max(
                calc_depth(self.message_graph[child_id])
                for child_id in node.children
                if child_id in self.message_graph
            )

        depth = calc_depth(root)

        # Normalize subject (remove Re:, Fwd:, etc.)
        subject = self._normalize_subject(root.subject)

        thread = EmailThread(
            thread_id=root.message_id,
            root_message_id=root.message_id,
            message_ids=message_ids,
            message_count=len(message_ids),
            participants=list(participants_map.values()),
            participant_count=len(participants_map),
            subject=subject,
            start_date=start_date,
            last_message_date=last_date,
            duration_days=duration_days,
            depth=depth,
            mailbox_id="",  # Set by caller
            reconstructed_at=datetime.utcnow(),
        )

        return thread

    def _normalize_subject(self, subject: str) -> str:
        """Remove Re:/Fwd: prefixes to get original subject."""
        import re
        # Remove common prefixes
        pattern = r'^(Re|RE|re|Fwd|FWD|fwd):\s*'
        while re.match(pattern, subject):
            subject = re.sub(pattern, '', subject, count=1).strip()
        return subject

    def calculate_response_times(
        self,
        thread: EmailThread,
        messages: Dict[str, EmailMessage],
    ) -> None:
        """Calculate response time statistics for thread."""
        response_times = []

        for message_id in thread.message_ids:
            if message_id not in self.message_graph:
                continue

            node = self.message_graph[message_id]
            if not node.parent_message_id or node.parent_message_id not in self.message_graph:
                continue

            parent = self.message_graph[node.parent_message_id]
            response_time_minutes = (node.date - parent.date).total_seconds() / 60
            response_times.append(response_time_minutes)

        if response_times:
            thread.total_response_time_minutes = sum(response_times)
            thread.average_response_time_minutes = sum(response_times) / len(response_times)
```

### Thread-Based Semantic Triples
```python
def extract_thread_triples(thread: EmailThread) -> List[SemanticTriple]:
    """Extract semantic triples from thread structure."""
    triples = []

    thread_uri = f"thread:{thread.thread_id}"

    # Thread type
    triples.append(SemanticTriple(
        subject=thread_uri,
        predicate="rdf:type",
        object="futurnal:EmailThread",
        extraction_method="thread_reconstruction",
    ))

    # Thread metadata
    triples.append(SemanticTriple(
        subject=thread_uri,
        predicate="thread:subject",
        object=thread.subject,
        extraction_method="thread_reconstruction",
    ))

    triples.append(SemanticTriple(
        subject=thread_uri,
        predicate="thread:messageCount",
        object=str(thread.message_count),
        extraction_method="thread_reconstruction",
    ))

    triples.append(SemanticTriple(
        subject=thread_uri,
        predicate="thread:participantCount",
        object=str(thread.participant_count),
        extraction_method="thread_reconstruction",
    ))

    # Thread messages
    for message_id in thread.message_ids:
        email_uri = f"email:{message_id}"
        triples.append(SemanticTriple(
            subject=email_uri,
            predicate="email:partOfThread",
            object=thread_uri,
            extraction_method="thread_reconstruction",
        ))

    # Participant relationships
    for participant in thread.participants:
        person_uri = f"person:{participant.email_address}"

        triples.append(SemanticTriple(
            subject=person_uri,
            predicate="person:participatedIn",
            object=thread_uri,
            extraction_method="thread_reconstruction",
        ))

        triples.append(SemanticTriple(
            subject=person_uri,
            predicate=f"person:threadRole",
            object=participant.role.value,
            extraction_method="thread_reconstruction",
        ))

    # Conversation flow (parent-child relationships)
    for message_id in thread.message_ids:
        if message_id not in thread_reconstructor.message_graph:
            continue

        node = thread_reconstructor.message_graph[message_id]
        if node.parent_message_id:
            triples.append(SemanticTriple(
                subject=f"email:{message_id}",
                predicate="email:inReplyTo",
                object=f"email:{node.parent_message_id}",
                extraction_method="thread_reconstruction",
            ))

    return triples
```

### Subject Evolution Tracking
```python
class SubjectEvolutionTracker:
    """Track how subject line evolves in a thread."""

    def analyze_subject_evolution(self, thread: EmailThread, messages: Dict[str, EmailMessage]) -> List[str]:
        """Analyze subject variations in thread."""
        variations = []
        seen_subjects = set()

        for message_id in thread.message_ids:
            if message_id not in messages:
                continue

            msg = messages[message_id]
            normalized = self._normalize_subject(msg.subject or "")

            if normalized not in seen_subjects:
                seen_subjects.add(normalized)
                variations.append(msg.subject or "")

        return variations

    def _normalize_subject(self, subject: str) -> str:
        """Normalize subject for comparison."""
        import re
        # Remove Re:/Fwd: and extra whitespace
        pattern = r'^(Re|RE|re|Fwd|FWD|fwd):\s*'
        while re.match(pattern, subject):
            subject = re.sub(pattern, '', subject, count=1).strip()
        return subject.lower().strip()
```

## Acceptance Criteria

- ✅ Threads reconstructed correctly from Message-ID/References/In-Reply-To
- ✅ Thread tree structure built with correct parent-child relationships
- ✅ Participants identified with roles (initiator, recipient, participant)
- ✅ Subject normalization removes Re:/Fwd: prefixes
- ✅ Response time statistics calculated (average, total)
- ✅ Thread depth and branch count calculated
- ✅ Semantic triples generated for threads and participants
- ✅ Thread graph stored in PKG with conversation relationships
- ✅ Orphan messages handled gracefully (missing parents)
- ✅ Out-of-order message arrival handled (parent arrives after child)

## Test Plan

### Unit Tests
- Message-ID graph construction
- Parent-child linking via In-Reply-To
- Parent-child linking via References
- Subject normalization
- Participant role assignment
- Response time calculation
- Thread depth calculation

### Integration Tests
- Multi-message thread reconstruction
- Branching thread reconstruction
- Out-of-order message handling
- Missing parent message handling
- Duplicate message handling
- Semantic triple generation

### Real-World Tests
- Gmail thread format
- Office 365 thread format
- Long threads (>50 messages)
- Deep threads (>10 levels)
- Wide threads (many branches)
- Cross-mailbox threads (partial views)

## Implementation Notes

### Thread ID Generation
```python
def generate_thread_id(root_message_id: str) -> str:
    """Use root message ID as thread ID."""
    return root_message_id
```

### Handling Missing Parents
```python
def handle_orphan_message(message: EmailMessage) -> None:
    """Handle message whose parent is not in mailbox."""
    # Create placeholder parent node
    if message.in_reply_to:
        placeholder = ThreadNode(
            message_id=message.in_reply_to,
            parent_message_id=None,
            date=message.date - timedelta(minutes=30),  # Estimate
            from_address="unknown@unknown",
            subject=message.subject or "",
        )
        self.message_graph[message.in_reply_to] = placeholder
```

### RFC 2822 Compliance
Per RFC 2822, the References header should contain:
1. Contents of parent's References header
2. Parent's Message-ID

This creates a chain from root to current message.

## Open Questions

- How to handle thread merges (two threads combined)?
- Should we detect thread splits (one thread becomes two)?
- How to handle cross-mailbox threads (partial views)?
- Should we use subject similarity for thread matching (in addition to headers)?
- How to handle very long threads (>100 messages)?
- Should we detect automated responses (out-of-office, bounces)?

## Dependencies
- EmailParser from task 04
- SemanticTripleExtractor from pipeline
- PKG storage for thread relationships


