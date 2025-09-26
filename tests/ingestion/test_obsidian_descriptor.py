"""Tests for Obsidian vault descriptor and registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from futurnal.ingestion.obsidian.descriptor import (
    DEFAULT_OBSIDIAN_IGNORE_RULES,
    ObsidianVaultDescriptor,
    VaultRegistry,
)


def _make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()
    return vault


def test_descriptor_id_determinism(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    d1 = ObsidianVaultDescriptor.from_path(vault)
    d2 = ObsidianVaultDescriptor.from_path(vault)
    assert d1.id == d2.id


def test_ignore_merge_with_file_and_extra(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    (vault / ".futurnalignore").write_text("custom/**\n#comment\n\n")
    d = ObsidianVaultDescriptor.from_path(vault, extra_ignores=["temp/**"])  # type: ignore[arg-type]
    for rule in DEFAULT_OBSIDIAN_IGNORE_RULES:
        assert rule in d.ignore_rules
    assert "custom/**" in d.ignore_rules
    assert "temp/**" in d.ignore_rules


def test_registry_crud_and_idempotent_register(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    vault = _make_vault(tmp_path)
    reg = VaultRegistry(registry_root=registry_dir)

    d1 = reg.register_path(vault, name="My Vault")
    # Running again should update, not duplicate
    d2 = reg.register_path(vault, name="My Vault")
    assert d1.id == d2.id
    assert len(list(registry_dir.glob("*.json"))) == 1

    # List and show
    items = reg.list()
    assert items and items[0].id == d1.id
    loaded = reg.get(d1.id)
    assert loaded.base_path == d1.base_path

    # Remove
    reg.remove(d1.id)
    assert not list(registry_dir.glob("*.json"))


def test_to_local_source_conversion(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    descriptor = ObsidianVaultDescriptor.from_path(vault, name="Test Vault")
    
    local_source = descriptor.to_local_source(
        max_workers=4,
        schedule="0 */6 * * *",
        priority="high",
    )
    
    assert local_source.name == "Test Vault"
    assert local_source.root_path == vault
    assert local_source.exclude == descriptor.ignore_rules
    assert local_source.max_workers == 4
    assert local_source.schedule == "0 */6 * * *"
    assert local_source.priority == "high"
    assert local_source.require_external_processing_consent is True
    assert local_source.external_processing_scope == "obsidian.external_processing"


def test_redaction_policy_with_title_patterns(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    descriptor = ObsidianVaultDescriptor.from_path(
        vault, 
        redact_title_patterns=["secret.*", "private-.*"]
    )
    
    policy = descriptor.build_redaction_policy()
    
    # Test sensitive title redaction
    sensitive_path = vault / "secret-document.md"
    redacted = policy.apply(sensitive_path)
    assert "secret-document" not in redacted.redacted
    
    # Test normal title
    normal_path = vault / "normal-note.md"
    redacted = policy.apply(normal_path)
    assert "normal-note" in redacted.redacted or ".md" in redacted.redacted


def test_network_mount_detection(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    descriptor = ObsidianVaultDescriptor.from_path(vault)
    
    # For local paths, should return None
    warning = descriptor.get_network_warning()
    assert warning is None  # Local tmp_path should not trigger warning


def test_duplicate_registration_preserves_created_at(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    vault = _make_vault(tmp_path)
    reg = VaultRegistry(registry_root=registry_dir)

    # First registration
    d1 = reg.register_path(vault, name="Original")
    original_created = d1.created_at
    
    # Second registration should preserve created_at
    d2 = reg.register_path(vault, name="Updated")
    assert d1.id == d2.id
    assert d2.created_at == original_created
    assert d2.name == "Updated"  # But update other fields
    assert d2.updated_at > original_created


