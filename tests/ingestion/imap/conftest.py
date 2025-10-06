"""Shared test fixtures and mock infrastructure for IMAP tests."""

from __future__ import annotations

import email
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import pytest


# ============================================================================
# Mock IMAP Server
# ============================================================================


class MockImapServer:
    """Mock IMAP server for testing.

    Simulates realistic IMAP protocol responses for testing without
    requiring a real IMAP server or network connection.
    """

    def __init__(
        self,
        provider: str = "generic",
        capabilities: Optional[List[bytes]] = None,
    ):
        """Initialize mock IMAP server.

        Args:
            provider: Provider type ("gmail", "office365", "generic")
            capabilities: IMAP capabilities to advertise
        """
        self.provider = provider
        self.messages: Dict[int, bytes] = {}
        self.uidvalidity = 1
        self.next_uid = 1
        self.modseq = 1
        self.selected_folder: Optional[str] = None

        # Default capabilities
        if capabilities is None:
            self.capabilities = [b"IMAP4rev1", b"IDLE", b"CONDSTORE"]
            if provider == "gmail":
                self.capabilities.extend([b"X-GM-EXT-1", b"UIDPLUS"])
            elif provider == "office365":
                self.capabilities.extend([b"UIDPLUS", b"MOVE"])
        else:
            self.capabilities = capabilities

        # Gmail-specific
        self.gmail_labels: Dict[int, List[str]] = {}
        self.gmail_thread_ids: Dict[int, str] = {}

        # Folder structure
        if provider == "gmail":
            self.folders = [
                "INBOX",
                "[Gmail]/All Mail",
                "[Gmail]/Sent Mail",
                "[Gmail]/Drafts",
                "[Gmail]/Trash",
            ]
        elif provider == "office365":
            self.folders = ["INBOX", "Sent Items", "Drafts", "Deleted Items", "Archive"]
        else:
            self.folders = ["INBOX", "Sent", "Drafts", "Trash"]

    def add_message(self, uid: int, raw_message: bytes, **metadata) -> None:
        """Add message to server.

        Args:
            uid: Message UID
            raw_message: Raw RFC822 message bytes
            **metadata: Additional metadata (labels, thread_id, etc.)
        """
        self.messages[uid] = raw_message

        # Gmail-specific metadata
        if self.provider == "gmail":
            if "labels" in metadata:
                self.gmail_labels[uid] = metadata["labels"]
            if "thread_id" in metadata:
                self.gmail_thread_ids[uid] = metadata["thread_id"]

        # Update next UID
        if uid >= self.next_uid:
            self.next_uid = uid + 1

    def simulate_new_message(self, raw_message: bytes, **metadata) -> int:
        """Simulate new message arrival.

        Args:
            raw_message: Raw RFC822 message bytes
            **metadata: Additional metadata

        Returns:
            Assigned UID
        """
        uid = self.next_uid
        self.next_uid += 1
        self.modseq += 1
        self.add_message(uid, raw_message, **metadata)
        return uid

    def simulate_deletion(self, uid: int) -> None:
        """Simulate message deletion.

        Args:
            uid: Message UID to delete
        """
        if uid in self.messages:
            del self.messages[uid]
            self.gmail_labels.pop(uid, None)
            self.gmail_thread_ids.pop(uid, None)
            self.modseq += 1

    def change_uidvalidity(self) -> None:
        """Simulate UIDVALIDITY change (e.g., mailbox migration).

        This forces a full resync as UIDs are no longer valid.
        """
        self.uidvalidity += 1
        self.messages.clear()
        self.gmail_labels.clear()
        self.gmail_thread_ids.clear()
        self.next_uid = 1
        self.modseq = 1

    def select_folder(self, folder: str) -> Dict[bytes, Any]:
        """Simulate SELECT command.

        Args:
            folder: Folder name to select

        Returns:
            Folder status information
        """
        self.selected_folder = folder
        return {
            b"UIDVALIDITY": self.uidvalidity,
            b"EXISTS": len(self.messages),
            b"RECENT": 0,
            b"UIDNEXT": self.next_uid,
            b"HIGHESTMODSEQ": self.modseq if b"CONDSTORE" in self.capabilities else None,
        }

    def search(self, criteria: List[str]) -> List[int]:
        """Simulate SEARCH command.

        Args:
            criteria: Search criteria

        Returns:
            List of matching UIDs
        """
        # Simple implementation: return all UIDs by default
        # Real tests can customize via monkeypatch
        return sorted(self.messages.keys())

    def fetch(self, uids: List[int], items: List[str]) -> Dict[int, Dict[bytes, Any]]:
        """Simulate FETCH command.

        Args:
            uids: UIDs to fetch
            items: Items to fetch (FLAGS, RFC822, etc.)

        Returns:
            Fetch results keyed by UID
        """
        results = {}
        for uid in uids:
            if uid not in self.messages:
                continue

            fetch_data = {
                b"UID": uid,
                b"SEQ": uid,  # Simplified: UID == sequence number
            }

            if "FLAGS" in items or "X-GM-LABELS" in items:
                fetch_data[b"FLAGS"] = []

            if "RFC822" in items or "BODY[]" in items:
                fetch_data[b"RFC822"] = self.messages[uid]
                fetch_data[b"BODY[]"] = self.messages[uid]

            if "MODSEQ" in items and b"CONDSTORE" in self.capabilities:
                fetch_data[b"MODSEQ"] = self.modseq

            # Gmail-specific
            if self.provider == "gmail":
                if uid in self.gmail_labels:
                    fetch_data[b"X-GM-LABELS"] = [
                        label.encode() for label in self.gmail_labels[uid]
                    ]
                if uid in self.gmail_thread_ids:
                    fetch_data[b"X-GM-THRID"] = self.gmail_thread_ids[uid].encode()

            results[uid] = fetch_data

        return results

    def idle_check(self, timeout: float = 0.1) -> List[tuple]:
        """Simulate IDLE command response.

        Args:
            timeout: Idle timeout in seconds

        Returns:
            List of IDLE events
        """
        # Return empty list (no new messages)
        # Tests can monkeypatch to simulate events
        return []

    def noop(self) -> None:
        """Simulate NOOP command (keep-alive)."""
        pass

    def get_capabilities(self) -> List[bytes]:
        """Get server capabilities."""
        return self.capabilities.copy()

    def list_folders(self) -> List[str]:
        """List available folders."""
        return self.folders.copy()


# ============================================================================
# Test Fixtures - Mock Server
# ============================================================================


@pytest.fixture
def mock_imap_server() -> MockImapServer:
    """Provide generic mock IMAP server."""
    return MockImapServer(provider="generic")


@pytest.fixture
def mock_gmail_server() -> MockImapServer:
    """Provide Gmail-specific mock IMAP server."""
    return MockImapServer(provider="gmail")


@pytest.fixture
def mock_office365_server() -> MockImapServer:
    """Provide Office365-specific mock IMAP server."""
    return MockImapServer(provider="office365")


# ============================================================================
# Test Fixtures - Sample Email Messages
# ============================================================================


@pytest.fixture
def sample_email_message() -> bytes:
    """Provide sample RFC822 email message."""
    msg = MIMEText("This is a test email body.")
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Subject"] = "Test Email"
    msg["Message-ID"] = "<test123@example.com>"
    msg["Date"] = "Mon, 1 Jan 2024 12:00:00 +0000"
    return msg.as_bytes()


@pytest.fixture
def gmail_threaded_messages() -> List[bytes]:
    """Provide Gmail-style threaded messages with proper headers.

    Returns:
        List of 3 messages forming a thread (A -> B -> C)
    """
    messages = []

    # Message A (root)
    msg_a = MIMEText("This is the first message.")
    msg_a["From"] = "alice@example.com"
    msg_a["To"] = "bob@example.com"
    msg_a["Subject"] = "Project Discussion"
    msg_a["Message-ID"] = "<msg-a@example.com>"
    msg_a["Date"] = "Mon, 1 Jan 2024 10:00:00 +0000"
    messages.append(msg_a.as_bytes())

    # Message B (reply to A)
    msg_b = MIMEText("This is a reply to the first message.")
    msg_b["From"] = "bob@example.com"
    msg_b["To"] = "alice@example.com"
    msg_b["Subject"] = "Re: Project Discussion"
    msg_b["Message-ID"] = "<msg-b@example.com>"
    msg_b["In-Reply-To"] = "<msg-a@example.com>"
    msg_b["References"] = "<msg-a@example.com>"
    msg_b["Date"] = "Mon, 1 Jan 2024 11:00:00 +0000"
    messages.append(msg_b.as_bytes())

    # Message C (reply to B)
    msg_c = MIMEText("This is another reply.")
    msg_c["From"] = "alice@example.com"
    msg_c["To"] = "bob@example.com"
    msg_c["Subject"] = "Re: Project Discussion"
    msg_c["Message-ID"] = "<msg-c@example.com>"
    msg_c["In-Reply-To"] = "<msg-b@example.com>"
    msg_c["References"] = "<msg-a@example.com> <msg-b@example.com>"
    msg_c["Date"] = "Mon, 1 Jan 2024 12:00:00 +0000"
    messages.append(msg_c.as_bytes())

    return messages


@pytest.fixture
def office365_messages() -> List[bytes]:
    """Provide Office365-formatted messages."""
    messages = []

    msg = MIMEText("Office365 test message.")
    msg["From"] = "user@contoso.com"
    msg["To"] = "colleague@contoso.com"
    msg["Subject"] = "Office365 Test"
    msg["Message-ID"] = "<office365-msg@contoso.com>"
    msg["Date"] = "Mon, 1 Jan 2024 12:00:00 +0000"
    # Office365-specific headers
    msg["X-MS-Exchange-Organization-RecordReviewCfmType"] = "0"
    messages.append(msg.as_bytes())

    return messages


@pytest.fixture
def email_with_attachment() -> bytes:
    """Provide email message with attachment."""
    msg = MIMEMultipart()
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Subject"] = "Email with Attachment"
    msg["Message-ID"] = "<attachment-test@example.com>"
    msg["Date"] = "Mon, 1 Jan 2024 12:00:00 +0000"

    # Add text body
    body = MIMEText("Please see attached document.")
    msg.attach(body)

    # Add attachment
    attachment = MIMEText("This is attachment content.", "plain")
    attachment.add_header("Content-Disposition", "attachment", filename="document.txt")
    msg.attach(attachment)

    return msg.as_bytes()


# ============================================================================
# Test Fixtures - Large Datasets for Performance Testing
# ============================================================================


@pytest.fixture
def large_mailbox_dataset(mock_imap_server: MockImapServer) -> MockImapServer:
    """Provide mock server with 10,000+ messages for performance testing.

    Args:
        mock_imap_server: Mock server to populate

    Returns:
        Populated mock server
    """
    # Add 10,000 simple messages
    for i in range(10000):
        msg = MIMEText(f"Test message {i}")
        msg["From"] = f"sender{i % 100}@example.com"
        msg["To"] = "recipient@example.com"
        msg["Subject"] = f"Test Message {i}"
        msg["Message-ID"] = f"<msg{i}@example.com>"
        msg["Date"] = datetime(2024, 1, 1, 0, 0, i % 86400).strftime(
            "%a, %d %b %Y %H:%M:%S +0000"
        )
        mock_imap_server.add_message(i + 1, msg.as_bytes())

    return mock_imap_server


@pytest.fixture
def ground_truth_threads() -> Dict[str, Any]:
    """Provide pre-analyzed thread structures for accuracy validation.

    Returns:
        Dictionary with thread structures and expected relationships
    """
    return {
        "simple_thread": {
            "messages": [
                {"id": "A", "parent": None},
                {"id": "B", "parent": "A"},
                {"id": "C", "parent": "B"},
            ],
            "expected_structure": {"A": ["B"], "B": ["C"], "C": []},
        },
        "branching_thread": {
            "messages": [
                {"id": "A", "parent": None},
                {"id": "B", "parent": "A"},
                {"id": "C", "parent": "A"},
                {"id": "D", "parent": "B"},
            ],
            "expected_structure": {"A": ["B", "C"], "B": ["D"], "C": [], "D": []},
        },
        "out_of_order": {
            "messages": [
                {"id": "C", "parent": "B"},
                {"id": "A", "parent": None},
                {"id": "B", "parent": "A"},
            ],
            "expected_structure": {"A": ["B"], "B": ["C"], "C": []},
        },
    }


# ============================================================================
# Test Fixtures - Metrics and Quality Gates
# ============================================================================


@pytest.fixture
def metrics_collector():
    """Provide fresh metrics collector for tests."""
    from futurnal.ingestion.imap.sync_metrics import ImapSyncMetricsCollector

    return ImapSyncMetricsCollector()


@pytest.fixture
def quality_gate_evaluator(metrics_collector):
    """Provide quality gate evaluator with metrics collector."""
    from futurnal.ingestion.imap.quality_gate import (
        ImapQualityGateEvaluator,
        ImapQualityGates,
    )

    config = ImapQualityGates()
    return ImapQualityGateEvaluator(config=config, metrics_collector=metrics_collector)
