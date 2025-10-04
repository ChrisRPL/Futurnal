"""Tests for IMAP mailbox descriptor and registry."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from futurnal.ingestion.imap.descriptor import (
    AuthMode,
    ConsentScope,
    ImapMailboxDescriptor,
    MailboxPrivacySettings,
    MailboxRegistry,
    PrivacyLevel,
    create_credential_id,
    generate_mailbox_id,
)


def _mailbox_id(email: str, host: str) -> str:
    return generate_mailbox_id(email, host)


def _create_descriptor(**overrides) -> ImapMailboxDescriptor:
    base_kwargs = {
        "email_address": "user@example.com",
        "imap_host": "imap.example.com",
        "imap_port": 993,
        "auth_mode": AuthMode.OAUTH2,
        "credential_id": "imap_cred_test",
        "privacy_settings": MailboxPrivacySettings(
            privacy_level=PrivacyLevel.STANDARD,
            required_consent_scopes=[ConsentScope.MAILBOX_ACCESS],
        ),
    }
    base_kwargs.update(overrides)
    descriptor = ImapMailboxDescriptor.from_registration(**base_kwargs)
    return descriptor.update(
        id=_mailbox_id(base_kwargs["email_address"], base_kwargs["imap_host"]),
        credential_id=create_credential_id(
            _mailbox_id(base_kwargs["email_address"], base_kwargs["imap_host"])
        ),
    )


def test_descriptor_id_determinism() -> None:
    desc1 = _create_descriptor()
    desc2 = _create_descriptor()
    assert desc1.id == desc2.id


def test_descriptor_folder_overlap_validation() -> None:
    with pytest.raises(ValueError):
        _create_descriptor(folders=["INBOX"], exclude_folders=["INBOX"])


def test_descriptor_to_local_source(tmp_path: Path) -> None:
    descriptor = _create_descriptor(name="Work Mail")
    workspace = tmp_path / "workspace"
    local_source = descriptor.to_local_source(workspace_root=workspace, schedule="0 */6 * * *", priority="high")
    assert local_source.name == "Work Mail"
    assert local_source.root_path == (workspace / descriptor.id)
    assert local_source.schedule == "0 */6 * * *"
    assert local_source.priority == "high"
    assert descriptor.id in str(local_source.root_path)


def test_registry_crud_and_idempotent_registration(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    registry = MailboxRegistry(registry_root=registry_dir)
    descriptor = _create_descriptor()

    saved = registry.add_or_update(descriptor)
    assert saved.id == descriptor.id
    assert (registry_dir / f"{descriptor.id}.json").exists()

    again = registry.add_or_update(descriptor)
    assert again.id == descriptor.id
    assert len(list(registry_dir.glob("*.json"))) == 1

    listed = registry.list()
    assert listed and listed[0].id == descriptor.id

    loaded = registry.get(descriptor.id)
    assert loaded.email_address == descriptor.email_address

    registry.remove(descriptor.id)
    assert not list(registry_dir.glob("*.json"))


def test_generate_mailbox_id_unique_for_different_hosts() -> None:
    id1 = generate_mailbox_id("user@example.com", "imap.example.com")
    id2 = generate_mailbox_id("user@example.com", "imap.other.com")
    assert id1 != id2


def test_privacy_settings_allow_configuration() -> None:
    settings = MailboxPrivacySettings(
        privacy_level=PrivacyLevel.PERMISSIVE,
        required_consent_scopes=[
            ConsentScope.MAILBOX_ACCESS,
            ConsentScope.EMAIL_CONTENT_ANALYSIS,
            ConsentScope.THREAD_RECONSTRUCTION,
        ],
        enable_sender_anonymization=False,
        enable_recipient_anonymization=False,
        enable_subject_redaction=True,
        redact_email_patterns=[r".*@test.com"],
        exclude_email_patterns=[r".*noreply.*"],
        privacy_subject_keywords=["secret"],
        audit_sync_events=False,
        audit_content_changes=True,
        retain_audit_days=120,
    )
    descriptor = _create_descriptor(privacy_settings=settings)
    assert descriptor.privacy_settings.privacy_level == PrivacyLevel.PERMISSIVE
    assert ConsentScope.THREAD_RECONSTRUCTION in descriptor.privacy_settings.required_consent_scopes


def test_sync_from_date_validation() -> None:
    now = datetime.utcnow()
    descriptor = _create_descriptor(sync_from_date=now)
    assert descriptor.sync_from_date == now


def test_max_message_age_days_validation() -> None:
    descriptor = _create_descriptor(max_message_age_days=30)
    assert descriptor.max_message_age_days == 30


def test_descriptor_serialization_roundtrip(tmp_path: Path) -> None:
    descriptor = _create_descriptor(name="Personal")
    path = tmp_path / "descriptor.json"
    path.write_text(json.dumps(descriptor.model_dump(mode="json"), indent=2))
    loaded = ImapMailboxDescriptor.model_validate(json.loads(path.read_text()))
    assert loaded.id == descriptor.id
    assert loaded.email_address == descriptor.email_address


