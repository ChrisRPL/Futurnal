"""Tests for email semantic triple extraction.

Tests cover:
- Email entity triple generation
- Person entity triple generation
- Threading relationship triples
- Participant relationship triples
- Metadata preservation
- Edge cases (missing fields, empty lists)
"""

from __future__ import annotations

from datetime import datetime

import pytest

from futurnal.ingestion.imap.email_parser import (
    AttachmentMetadata,
    EmailAddress,
    EmailMessage,
)
from futurnal.ingestion.imap.email_triples import extract_email_triples


def _create_test_email_message(**overrides) -> EmailMessage:
    """Create a test EmailMessage with default values."""
    defaults = {
        "message_id": "test123@example.com",
        "uid": 1,
        "folder": "INBOX",
        "subject": "Test Subject",
        "from_address": EmailAddress(
            address="sender@example.com", display_name="John Doe"
        ),
        "to_addresses": [
            EmailAddress(address="recipient@example.com", display_name="Jane Smith")
        ],
        "cc_addresses": [],
        "bcc_addresses": [],
        "reply_to_addresses": [],
        "date": datetime(2024, 1, 15, 10, 30, 0),
        "in_reply_to": None,
        "references": [],
        "body_plain": "Test email body",
        "body_html": None,
        "body_normalized": "Test email body",
        "size_bytes": 1024,
        "flags": [],
        "labels": [],
        "attachments": [],
        "contains_sensitive_keywords": False,
        "privacy_classification": "standard",
        "retrieved_at": datetime(2024, 1, 15, 10, 35, 0),
        "mailbox_id": "test-mailbox",
    }
    defaults.update(overrides)
    return EmailMessage(**defaults)


def test_extract_basic_email_triples():
    """Test extraction of basic email entity triples."""
    email_msg = _create_test_email_message()
    triples = extract_email_triples(email_msg)

    # Convert to dict for easier testing
    triples_dict = {(t.subject, t.predicate, t.object) for t in triples}

    # Email type triple
    assert (
        "futurnal:email/test123@example.com",
        "rdf:type",
        "futurnal:Email",
    ) in triples_dict

    # Subject triple
    assert (
        "futurnal:email/test123@example.com",
        "email:subject",
        "Test Subject",
    ) in triples_dict

    # Date triple
    assert (
        "futurnal:email/test123@example.com",
        "email:sentDate",
        "2024-01-15T10:30:00",
    ) in triples_dict

    # Folder triple
    assert (
        "futurnal:email/test123@example.com",
        "email:folder",
        "INBOX",
    ) in triples_dict


def test_extract_sender_triples():
    """Test extraction of sender person triples."""
    email_msg = _create_test_email_message(
        from_address=EmailAddress(
            address="john@example.com", display_name="John Doe"
        )
    )
    triples = extract_email_triples(email_msg)

    triples_dict = {(t.subject, t.predicate, t.object) for t in triples}

    # Email from person
    assert (
        "futurnal:email/test123@example.com",
        "email:from",
        "futurnal:person/john@example.com",
    ) in triples_dict

    # Person type
    assert (
        "futurnal:person/john@example.com",
        "rdf:type",
        "futurnal:Person",
    ) in triples_dict

    # Person email address
    assert (
        "futurnal:person/john@example.com",
        "person:emailAddress",
        "john@example.com",
    ) in triples_dict

    # Person display name
    assert (
        "futurnal:person/john@example.com",
        "person:displayName",
        "John Doe",
    ) in triples_dict


def test_extract_sender_without_display_name():
    """Test sender triples when display name is missing."""
    email_msg = _create_test_email_message(
        from_address=EmailAddress(address="john@example.com", display_name=None),
        to_addresses=[EmailAddress(address="no-display@example.com", display_name=None)],
    )
    triples = extract_email_triples(email_msg)

    triples_dict = {(t.subject, t.predicate, t.object) for t in triples}

    # Should have email address but not display name
    assert (
        "futurnal:person/john@example.com",
        "person:emailAddress",
        "john@example.com",
    ) in triples_dict

    # No display name triple for anyone (sender or recipient without display names)
    display_name_triples = [
        t for t in triples if t.predicate == "person:displayName"
    ]
    assert len(display_name_triples) == 0


def test_extract_recipient_triples():
    """Test extraction of recipient triples."""
    email_msg = _create_test_email_message(
        to_addresses=[
            EmailAddress(address="alice@example.com", display_name="Alice"),
            EmailAddress(address="bob@example.com", display_name="Bob"),
        ]
    )
    triples = extract_email_triples(email_msg)

    triples_dict = {(t.subject, t.predicate, t.object) for t in triples}

    # Email to recipients
    assert (
        "futurnal:email/test123@example.com",
        "email:to",
        "futurnal:person/alice@example.com",
    ) in triples_dict

    assert (
        "futurnal:email/test123@example.com",
        "email:to",
        "futurnal:person/bob@example.com",
    ) in triples_dict

    # Recipient person entities
    assert (
        "futurnal:person/alice@example.com",
        "person:displayName",
        "Alice",
    ) in triples_dict

    assert (
        "futurnal:person/bob@example.com",
        "person:displayName",
        "Bob",
    ) in triples_dict


def test_extract_cc_triples():
    """Test extraction of Cc recipient triples."""
    email_msg = _create_test_email_message(
        cc_addresses=[
            EmailAddress(address="cc1@example.com"),
            EmailAddress(address="cc2@example.com"),
        ]
    )
    triples = extract_email_triples(email_msg)

    triples_dict = {(t.subject, t.predicate, t.object) for t in triples}

    # Email cc recipients
    assert (
        "futurnal:email/test123@example.com",
        "email:cc",
        "futurnal:person/cc1@example.com",
    ) in triples_dict

    assert (
        "futurnal:email/test123@example.com",
        "email:cc",
        "futurnal:person/cc2@example.com",
    ) in triples_dict


def test_extract_threading_triples():
    """Test extraction of threading relationship triples."""
    email_msg = _create_test_email_message(
        in_reply_to="parent123@example.com",
        references=["thread1@example.com", "thread2@example.com", "parent123@example.com"],
    )
    triples = extract_email_triples(email_msg)

    triples_dict = {(t.subject, t.predicate, t.object) for t in triples}

    # In-Reply-To triple
    assert (
        "futurnal:email/test123@example.com",
        "email:inReplyTo",
        "futurnal:email/parent123@example.com",
    ) in triples_dict

    # References triples
    assert (
        "futurnal:email/test123@example.com",
        "email:references",
        "futurnal:email/thread1@example.com",
    ) in triples_dict

    assert (
        "futurnal:email/test123@example.com",
        "email:references",
        "futurnal:email/thread2@example.com",
    ) in triples_dict

    assert (
        "futurnal:email/test123@example.com",
        "email:references",
        "futurnal:email/parent123@example.com",
    ) in triples_dict

    # Thread position triples
    position_triples = [
        t for t in triples if t.predicate == "email:threadPosition"
    ]
    assert len(position_triples) == 3


def test_extract_threading_without_in_reply_to():
    """Test threading triples when only References is present."""
    email_msg = _create_test_email_message(
        in_reply_to=None,
        references=["thread1@example.com", "thread2@example.com"],
    )
    triples = extract_email_triples(email_msg)

    # Should have references but not in-reply-to
    in_reply_to_triples = [
        t for t in triples if t.predicate == "email:inReplyTo"
    ]
    assert len(in_reply_to_triples) == 0

    references_triples = [
        t for t in triples if t.predicate == "email:references"
    ]
    assert len(references_triples) == 2


def test_extract_attachment_triples():
    """Test extraction of attachment-related triples."""
    email_msg = _create_test_email_message(
        attachments=[
            AttachmentMetadata(
                filename="doc.pdf",
                content_type="application/pdf",
                size_bytes=1024,
                part_id="1",
            ),
            AttachmentMetadata(
                filename="image.png",
                content_type="image/png",
                size_bytes=2048,
                part_id="2",
            ),
        ]
    )
    triples = extract_email_triples(email_msg)

    triples_dict = {(t.subject, t.predicate, t.object) for t in triples}

    # Has attachments triple
    assert (
        "futurnal:email/test123@example.com",
        "email:hasAttachments",
        "true",
    ) in triples_dict

    # Attachment count triple
    assert (
        "futurnal:email/test123@example.com",
        "email:attachmentCount",
        "2",
    ) in triples_dict


def test_no_attachment_triples_when_empty():
    """Test no attachment triples when email has no attachments."""
    email_msg = _create_test_email_message(attachments=[])
    triples = extract_email_triples(email_msg)

    attachment_triples = [
        t
        for t in triples
        if t.predicate in ("email:hasAttachments", "email:attachmentCount")
    ]
    assert len(attachment_triples) == 0


def test_extract_sensitive_keyword_triple():
    """Test extraction of sensitive keyword triple."""
    email_msg = _create_test_email_message(
        contains_sensitive_keywords=True,
        privacy_classification="sensitive",
    )
    triples = extract_email_triples(email_msg)

    triples_dict = {(t.subject, t.predicate, t.object) for t in triples}

    # Contains sensitive keywords triple
    assert (
        "futurnal:email/test123@example.com",
        "email:containsSensitiveKeywords",
        "true",
    ) in triples_dict


def test_no_sensitive_keyword_triple_when_false():
    """Test no sensitive keyword triple when not flagged."""
    email_msg = _create_test_email_message(
        contains_sensitive_keywords=False
    )
    triples = extract_email_triples(email_msg)

    sensitive_triples = [
        t for t in triples if t.predicate == "email:containsSensitiveKeywords"
    ]
    assert len(sensitive_triples) == 0


def test_extract_email_without_subject():
    """Test triple extraction when subject is missing."""
    email_msg = _create_test_email_message(subject=None)
    triples = extract_email_triples(email_msg)

    # Should not have subject triple
    subject_triples = [t for t in triples if t.predicate == "email:subject"]
    assert len(subject_triples) == 0


def test_triple_metadata_fields():
    """Test that triples have correct metadata fields."""
    email_msg = _create_test_email_message()
    triples = extract_email_triples(email_msg)

    # All triples should have required metadata
    for triple in triples:
        assert triple.source_element_id == "test123@example.com"
        assert triple.extraction_method == "email_metadata"
        assert triple.confidence == 1.0  # Default confidence
        assert triple.created_at is not None


def test_email_uri_creation():
    """Test email URI creation handles special characters."""
    email_msg = _create_test_email_message(
        message_id="<complex.id+test@example.com>"
    )
    triples = extract_email_triples(email_msg)

    # Find the email type triple
    type_triple = next(t for t in triples if t.predicate == "rdf:type")

    # URI should clean up angle brackets
    assert type_triple.subject == "futurnal:email/complex.id+test@example.com"
    assert "<" not in type_triple.subject
    assert ">" not in type_triple.subject


def test_person_uri_creation():
    """Test person URI creation normalizes email addresses."""
    email_msg = _create_test_email_message(
        from_address=EmailAddress(
            address="John.Doe@EXAMPLE.COM", display_name="John Doe"
        ),
        to_addresses=[],  # No recipients to avoid confusion
    )
    triples = extract_email_triples(email_msg)

    # Find person triples for the sender
    person_triples = [
        t for t in triples if "futurnal:person/john.doe@example.com" in t.subject
    ]

    # Should have at least one triple for the sender
    assert len(person_triples) > 0

    # All URIs should be lowercased
    for triple in person_triples:
        assert triple.subject == "futurnal:person/john.doe@example.com"


def test_full_email_triple_extraction():
    """Test complete triple extraction with all features."""
    email_msg = _create_test_email_message(
        message_id="full-test@example.com",
        subject="Project Discussion",
        from_address=EmailAddress(
            address="alice@example.com", display_name="Alice Johnson"
        ),
        to_addresses=[
            EmailAddress(address="bob@example.com", display_name="Bob Smith"),
            EmailAddress(address="charlie@example.com"),
        ],
        cc_addresses=[EmailAddress(address="manager@example.com")],
        date=datetime(2024, 3, 20, 14, 30, 0),
        folder="INBOX/Work",
        in_reply_to="previous@example.com",
        references=["thread1@example.com", "previous@example.com"],
        attachments=[
            AttachmentMetadata(
                filename="report.pdf",
                content_type="application/pdf",
                size_bytes=1024,
                part_id="1",
            )
        ],
        contains_sensitive_keywords=True,
    )
    triples = extract_email_triples(email_msg)

    # Verify we have a comprehensive set of triples
    assert len(triples) > 20  # Should have many triples for full email

    # Check for key categories
    email_triples = [t for t in triples if "futurnal:email/" in t.subject]
    person_triples = [t for t in triples if "futurnal:person/" in t.subject]

    assert len(email_triples) > 10
    assert len(person_triples) > 5

    # Verify extraction method
    assert all(t.extraction_method == "email_metadata" for t in triples)

    # Verify source element ID
    assert all(t.source_element_id == "full-test@example.com" for t in triples)


def test_extract_multiple_participants():
    """Test extraction creates distinct person entities for all participants."""
    email_msg = _create_test_email_message(
        from_address=EmailAddress(address="sender@example.com"),
        to_addresses=[
            EmailAddress(address="to1@example.com"),
            EmailAddress(address="to2@example.com"),
            EmailAddress(address="to3@example.com"),
        ],
        cc_addresses=[
            EmailAddress(address="cc1@example.com"),
            EmailAddress(address="cc2@example.com"),
        ],
    )
    triples = extract_email_triples(email_msg)

    # Extract unique person URIs
    person_uris = set()
    for triple in triples:
        if "futurnal:person/" in triple.subject:
            person_uris.add(triple.subject)

    # Should have 6 distinct person entities (1 sender + 3 to + 2 cc)
    assert len(person_uris) == 6


def test_triples_are_serializable():
    """Test that extracted triples can be converted to dict."""
    email_msg = _create_test_email_message()
    triples = extract_email_triples(email_msg)

    # All triples should be convertible to dict
    for triple in triples:
        triple_dict = triple.to_dict()
        assert isinstance(triple_dict, dict)
        assert "subject" in triple_dict
        assert "predicate" in triple_dict
        assert "object" in triple_dict
        assert "extraction_method" in triple_dict


def test_empty_references_list():
    """Test handling of empty references list."""
    email_msg = _create_test_email_message(references=[])
    triples = extract_email_triples(email_msg)

    # Should not have any references triples
    references_triples = [
        t for t in triples if t.predicate == "email:references"
    ]
    assert len(references_triples) == 0
