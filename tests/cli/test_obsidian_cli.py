"""CLI tests for Obsidian vault management."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from futurnal.cli.local_sources import app


runner = CliRunner()


def _make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()
    # Add a test note to avoid empty vault warnings in tests
    (vault / "test-note.md").write_text("# Test Note\n\nThis is a test note.")
    return vault


def test_obsidian_add_list_show_remove(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    registry = tmp_path / "registry"

    # add
    add_res = runner.invoke(
        app,
        [
            "obsidian",
            "add",
            "--path",
            str(vault),
            "--name",
            "Notes",
            "--json",
            "--registry-path",
            str(registry),
        ],
    )
    assert add_res.exit_code == 0
    payload = json.loads(add_res.output)
    vault_id = payload["id"]
    assert (registry / f"{vault_id}.json").exists()

    # list
    list_res = runner.invoke(app, ["obsidian", "list", "--json", "--registry-path", str(registry)])
    assert list_res.exit_code == 0
    items = json.loads(list_res.output)
    assert items and items[0]["id"] == vault_id

    # show
    show_res = runner.invoke(app, ["obsidian", "show", vault_id, "--registry-path", str(registry)])
    assert show_res.exit_code == 0
    shown = json.loads(show_res.output)
    assert shown["id"] == vault_id

    # remove
    rm_res = runner.invoke(app, ["obsidian", "remove", vault_id, "--yes", "--registry-path", str(registry)])
    assert rm_res.exit_code == 0
    assert not list(registry.glob("*.json"))


def test_obsidian_to_local_source_conversion(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    registry = tmp_path / "registry"

    # Add vault
    add_res = runner.invoke(
        app,
        [
            "obsidian", "add",
            "--path", str(vault),
            "--name", "Test Vault",
            "--json",
            "--registry-path", str(registry),
        ],
    )
    assert add_res.exit_code == 0
    payload = json.loads(add_res.output)
    vault_id = payload["id"]

    # Convert to local source
    convert_res = runner.invoke(
        app,
        [
            "obsidian", "to-local-source", vault_id,
            "--registry-path", str(registry),
            "--max-workers", "2",
            "--schedule", "0 */12 * * *",
            "--priority", "high",
        ],
    )
    assert convert_res.exit_code == 0
    local_source = json.loads(convert_res.output)
    assert local_source["name"] == "Test Vault"
    assert local_source["max_workers"] == 2
    assert local_source["schedule"] == "0 */12 * * *"
    assert local_source["priority"] == "high"
    assert local_source["require_external_processing_consent"] is True


def test_obsidian_network_warning_display(tmp_path: Path, monkeypatch) -> None:
    vault = _make_vault(tmp_path)
    registry = tmp_path / "registry"

    # Mock network mount detection to always return a warning
    def mock_detect_network_mount(path):
        return "Test warning: network mount detected"

    monkeypatch.setattr(
        "futurnal.ingestion.obsidian.descriptor._detect_network_mount",
        mock_detect_network_mount
    )

    # Add vault - should show warning
    add_res = runner.invoke(
        app,
        [
            "obsidian", "add",
            "--path", str(vault),
            "--registry-path", str(registry),
        ],
    )
    assert add_res.exit_code == 0
    assert "WARNING: Test warning: network mount detected" in add_res.stderr


def test_obsidian_add_with_redact_title_patterns(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    registry = tmp_path / "registry"

    # Add vault with redact-title patterns
    add_res = runner.invoke(
        app,
        [
            "obsidian", "add",
            "--path", str(vault),
            "--name", "Test Vault",
            "--redact-title", "secret.*,private-.*",
            "--json",
            "--registry-path", str(registry),
        ],
    )
    assert add_res.exit_code == 0
    payload = json.loads(add_res.output)
    vault_id = payload["id"]

    # Verify the redact patterns were stored
    descriptor_path = registry / f"{vault_id}.json"
    descriptor_data = json.loads(descriptor_path.read_text())
    assert descriptor_data["redact_title_patterns"] == ["secret.*", "private-.*"]

    # Test the redaction policy works
    from futurnal.ingestion.obsidian.descriptor import ObsidianVaultDescriptor
    descriptor = ObsidianVaultDescriptor.model_validate(descriptor_data)
    policy = descriptor.build_redaction_policy()

    # Test sensitive title redaction
    sensitive_path = vault / "secret-document.md"
    redacted = policy.apply(sensitive_path)
    assert "secret-document" not in redacted.redacted

    # Test normal title
    normal_path = vault / "normal-note.md"
    redacted = policy.apply(normal_path)
    assert "normal-note" in redacted.redacted or ".md" in redacted.redacted


def test_obsidian_add_with_json_flag(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    registry = tmp_path / "registry"

    # Test with --json flag
    add_res = runner.invoke(
        app,
        [
            "obsidian", "add",
            "--path", str(vault),
            "--name", "Test Vault",
            "--json",
            "--registry-path", str(registry),
        ],
    )
    assert add_res.exit_code == 0
    # Should output JSON when --json is provided
    payload = json.loads(add_res.output)
    assert payload["name"] == "Test Vault"


def test_obsidian_add_without_json_flag(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    registry = tmp_path / "registry"

    # Test without --json flag
    add_res = runner.invoke(
        app,
        [
            "obsidian", "add",
            "--path", str(vault),
            "--name", "Test Vault",
            "--registry-path", str(registry),
        ],
    )
    assert add_res.exit_code == 0
    # Should show user-friendly output with next steps
    assert "✅ Registered Obsidian vault" in add_res.output
    assert "📋 Next Steps:" in add_res.output
    assert "futurnal sources obsidian to-local-source" in add_res.output


def test_obsidian_remove_with_confirmation(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    registry = tmp_path / "registry"

    # Add vault first
    add_res = runner.invoke(
        app,
        [
            "obsidian", "add",
            "--path", str(vault),
            "--name", "Test Vault",
            "--json",
            "--registry-path", str(registry),
        ],
    )
    assert add_res.exit_code == 0
    payload = json.loads(add_res.output)
    vault_id = payload["id"]

    # Test remove with --yes flag
    remove_res = runner.invoke(
        app,
        [
            "obsidian", "remove", vault_id,
            "--yes",
            "--registry-path", str(registry),
        ],
    )
    assert remove_res.exit_code == 0
    assert "✅ Removed Obsidian vault" in remove_res.output


def test_top_level_add_obsidian_command(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    
    # Test the correct command structure: futurnal sources add obsidian
    add_res = runner.invoke(
        app,
        [
            "add", "obsidian",  # This is the required structure
            "--path", str(vault),
            "--name", "Test Vault",
            "--json",
        ],
    )
    assert add_res.exit_code == 0
    payload = json.loads(add_res.output)
    assert payload["name"] == "Test Vault"
    
    # Test without --json flag (user-friendly output)
    vault2_dir = tmp_path / "vault2"
    vault2_dir.mkdir()
    vault2 = _make_vault(vault2_dir)
    add_res2 = runner.invoke(
        app,
        [
            "add", "obsidian",
            "--path", str(vault2),
            "--name", "Test Vault 2",
        ],
    )
    assert add_res2.exit_code == 0
    assert "✅ Registered Obsidian vault" in add_res2.output
    assert "📋 Next Steps:" in add_res2.output
    
    # Test with redact-title patterns
    vault3_dir = tmp_path / "vault3"
    vault3_dir.mkdir()
    vault3 = _make_vault(vault3_dir)
    add_res3 = runner.invoke(
        app,
        [
            "add", "obsidian",
            "--path", str(vault3),
            "--name", "Secret Vault",
            "--redact-title", "secret.*,private-.*",
            "--json",
        ],
    )
    assert add_res3.exit_code == 0
    payload3 = json.loads(add_res3.output)
    assert payload3["redact_title_patterns"] == ["secret.*", "private-.*"]
    
    # Test unsupported source type
    unsupported_res = runner.invoke(
        app,
        [
            "add", "unsupported",
            "--path", str(vault),
        ],
    )
    assert unsupported_res.exit_code == 1
    assert "❌ Unsupported source type: unsupported" in unsupported_res.output
    assert "Supported types: obsidian" in unsupported_res.output


def test_top_level_remove_and_inspect(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    
    # Use default registry for this test
    add_res = runner.invoke(
        app,
        [
            "obsidian", "add",
            "--path", str(vault),
            "--name", "Test Vault",
            "--json",
        ],
    )
    assert add_res.exit_code == 0
    payload = json.loads(add_res.output)
    vault_id = payload["id"]

    # Test top-level inspect
    inspect_res = runner.invoke(app, ["inspect", vault_id])
    assert inspect_res.exit_code == 0
    inspect_payload = json.loads(inspect_res.output)
    assert inspect_payload["id"] == vault_id

    # Test top-level remove
    remove_res = runner.invoke(app, ["remove", vault_id, "--yes"])
    assert remove_res.exit_code == 0
    assert "✅ Removed Obsidian vault" in remove_res.output


def test_list_type_obsidian_filter(tmp_path: Path) -> None:
    """Test the 'futurnal sources list --type obsidian' command."""
    vault1_dir = tmp_path / "vault1"
    vault1_dir.mkdir()
    vault1 = _make_vault(vault1_dir)
    # Add content to avoid empty vault warning
    (vault1 / "note1.md").write_text("# Test Note 1")
    
    vault2_dir = tmp_path / "vault2" 
    vault2_dir.mkdir()
    vault2 = _make_vault(vault2_dir)
    # Add content to avoid empty vault warning
    (vault2 / "note2.md").write_text("# Test Note 2")
    
    registry = tmp_path / "registry"
    
    # Add two vaults
    add_res1 = runner.invoke(
        app,
        [
            "obsidian", "add",
            "--path", str(vault1),
            "--name", "Test Vault 1",
            "--json",
            "--registry-path", str(registry),
        ],
    )
    assert add_res1.exit_code == 0
    
    add_res2 = runner.invoke(
        app,
        [
            "obsidian", "add", 
            "--path", str(vault2),
            "--name", "Test Vault 2",
            "--json",
            "--registry-path", str(registry),
        ],
    )
    assert add_res2.exit_code == 0
    
    # Test list with --source-type obsidian filter
    list_res = runner.invoke(
        app,
        ["list", "--source-type", "obsidian", "--config-path", str(tmp_path / "sources.json")],
    )
    assert list_res.exit_code == 0
    
    # Since we're using a custom registry, we need to test the obsidian-specific list
    obsidian_list_res = runner.invoke(
        app,
        ["obsidian", "list", "--registry-path", str(registry)],
    )
    assert obsidian_list_res.exit_code == 0
    
    # Get vault IDs from the add commands to verify they appear in the list
    # Note: Output might contain warnings, so extract just the JSON part
    import re
    vault1_json_match = re.search(r'(\{.*\})', add_res1.output, re.DOTALL)
    vault2_json_match = re.search(r'(\{.*\})', add_res2.output, re.DOTALL)
    
    assert vault1_json_match, f"No JSON found in add_res1.output: {add_res1.output}"
    assert vault2_json_match, f"No JSON found in add_res2.output: {add_res2.output}"
    
    vault1_data = json.loads(vault1_json_match.group(1))
    vault2_data = json.loads(vault2_json_match.group(1))
    vault1_id = vault1_data["id"]
    vault2_id = vault2_data["id"]
    
    # Check that both vault IDs are in the output (paths are redacted for privacy)
    assert vault1_id in obsidian_list_res.output
    assert vault2_id in obsidian_list_res.output


def test_empty_vault_warning_display(tmp_path: Path) -> None:
    """Test that empty vault warnings are displayed properly."""
    vault = tmp_path / "empty_vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()
    # Note: Not creating any .md files to make it empty
    
    registry = tmp_path / "registry"
    
    # Add empty vault - should show warning
    add_res = runner.invoke(
        app,
        [
            "obsidian", "add",
            "--path", str(vault),
            "--name", "Empty Vault",
            "--registry-path", str(registry),
        ],
    )
    assert add_res.exit_code == 0
    assert "WARNING: Vault appears to be empty" in add_res.stderr
    assert "✅ Registered Obsidian vault" in add_res.output


def test_comprehensive_end_to_end_workflow(tmp_path: Path) -> None:
    """Test complete workflow: add → list → inspect → remove."""
    vault = _make_vault(tmp_path)
    # Add some content to avoid empty vault warning
    (vault / "note1.md").write_text("# Test Note 1")
    (vault / "note2.md").write_text("# Test Note 2")
    
    registry = tmp_path / "registry"
    
    # Step 1: Add vault
    add_res = runner.invoke(
        app,
        [
            "obsidian", "add",
            "--path", str(vault),
            "--name", "Workflow Test Vault",
            "--icon", "📚",
            "--redact-title", "secret.*",
            "--json",
            "--registry-path", str(registry),
        ],
    )
    assert add_res.exit_code == 0
    payload = json.loads(add_res.output)
    vault_id = payload["id"]
    assert payload["name"] == "Workflow Test Vault"
    assert payload["icon"] == "📚"
    assert payload["redact_title_patterns"] == ["secret.*"]
    
    # Step 2: List vaults (should include our vault)
    list_res = runner.invoke(
        app,
        ["obsidian", "list", "--json", "--registry-path", str(registry)],
    )
    assert list_res.exit_code == 0
    list_items = json.loads(list_res.output)
    assert len(list_items) == 1
    assert list_items[0]["id"] == vault_id
    assert list_items[0]["name"] == "Workflow Test Vault"
    
    # Step 3: Inspect vault
    inspect_res = runner.invoke(
        app,
        ["obsidian", "show", vault_id, "--registry-path", str(registry)],
    )
    assert inspect_res.exit_code == 0
    inspect_data = json.loads(inspect_res.output)
    assert inspect_data["id"] == vault_id
    assert inspect_data["name"] == "Workflow Test Vault"
    assert inspect_data["icon"] == "📚"
    
    # Also test top-level inspect command (will fail with custom registry, so just test that it handles the error gracefully)
    inspect_top_res = runner.invoke(app, ["inspect", vault_id])
    # With custom registry, this should fail gracefully with appropriate error message
    assert inspect_top_res.exit_code == 1
    assert "not found" in inspect_top_res.output.lower()
    
    # Step 4: Remove vault
    remove_res = runner.invoke(
        app,
        ["obsidian", "remove", vault_id, "--yes", "--registry-path", str(registry)],
    )
    assert remove_res.exit_code == 0
    assert "✅ Removed Obsidian vault" in remove_res.output
    
    # Step 5: Verify vault is gone
    final_list_res = runner.invoke(
        app,
        ["obsidian", "list", "--json", "--registry-path", str(registry)],
    )
    assert final_list_res.exit_code == 0
    final_list_items = json.loads(final_list_res.output)
    assert len(final_list_items) == 0


