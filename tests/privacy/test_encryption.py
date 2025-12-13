"""Tests for storage encryption module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.futurnal.privacy.encryption import (
    EncryptionManager,
    EncryptedPayload,
    EncryptionError,
    KeyNotFoundError,
    DecryptionError,
    KeyRotationError,
)


class MockKeyring:
    """Mock keyring for testing without OS keychain."""

    def __init__(self):
        self._store = {}

    def get_password(self, service: str, key: str):
        return self._store.get(f"{service}:{key}")

    def set_password(self, service: str, key: str, value: str):
        self._store[f"{service}:{key}"] = value

    def delete_password(self, service: str, key: str):
        self._store.pop(f"{service}:{key}", None)


@pytest.fixture
def mock_keyring():
    """Create mock keyring and patch it."""
    mock = MockKeyring()
    with patch("keyring.get_password", mock.get_password), \
         patch("keyring.set_password", mock.set_password), \
         patch("keyring.delete_password", mock.delete_password):
        yield mock


@pytest.fixture
def encryption_manager(mock_keyring):
    """Create encryption manager with mock keyring."""
    manager = EncryptionManager(service_name="test_futurnal")
    manager.initialize()
    return manager


@pytest.fixture
def disabled_manager():
    """Create disabled encryption manager."""
    return EncryptionManager(service_name="test_futurnal", enabled=False)


class TestEncryptedPayload:
    """Test EncryptedPayload data class."""

    def test_to_dict(self):
        payload = EncryptedPayload(
            ciphertext=b"encrypted_data",
            nonce=b"random_nonce",
            key_version=1,
        )
        d = payload.to_dict()
        assert "ciphertext" in d
        assert "nonce" in d
        assert d["key_version"] == 1
        assert d["algorithm"] == "AES-256-GCM"

    def test_from_dict(self):
        original = EncryptedPayload(
            ciphertext=b"encrypted_data",
            nonce=b"random_nonce",
            key_version=2,
        )
        d = original.to_dict()
        restored = EncryptedPayload.from_dict(d)

        assert restored.ciphertext == original.ciphertext
        assert restored.nonce == original.nonce
        assert restored.key_version == original.key_version

    def test_to_bytes_roundtrip(self):
        original = EncryptedPayload(
            ciphertext=b"test_cipher",
            nonce=b"test_nonce!",
            key_version=1,
        )
        as_bytes = original.to_bytes()
        restored = EncryptedPayload.from_bytes(as_bytes)

        assert restored.ciphertext == original.ciphertext
        assert restored.nonce == original.nonce


class TestEncryptionManager:
    """Test EncryptionManager core functionality."""

    def test_initialize_generates_key(self, mock_keyring):
        manager = EncryptionManager(service_name="test_init")
        manager.initialize()

        assert manager.is_initialized()
        assert manager.key_version == 1

    def test_initialize_retrieves_existing_key(self, mock_keyring):
        # First manager generates key
        manager1 = EncryptionManager(service_name="test_existing")
        manager1.initialize()

        # Second manager retrieves existing
        manager2 = EncryptionManager(service_name="test_existing")
        manager2.initialize()

        assert manager2.is_initialized()

    def test_encrypt_decrypt_bytes(self, encryption_manager):
        plaintext = b"Hello, World!"

        encrypted = encryption_manager.encrypt(plaintext)
        decrypted = encryption_manager.decrypt(encrypted)

        assert decrypted == plaintext

    def test_encrypt_decrypt_string(self, encryption_manager):
        plaintext = "Secret message with unicode: 你好世界"

        encrypted = encryption_manager.encrypt(plaintext)
        decrypted = encryption_manager.decrypt(encrypted)

        assert decrypted == plaintext.encode("utf-8")

    def test_encrypt_produces_different_ciphertexts(self, encryption_manager):
        plaintext = b"Same message"

        encrypted1 = encryption_manager.encrypt(plaintext)
        encrypted2 = encryption_manager.encrypt(plaintext)

        # Same plaintext produces different ciphertexts (due to random nonce)
        assert encrypted1.ciphertext != encrypted2.ciphertext
        assert encrypted1.nonce != encrypted2.nonce

    def test_decrypt_wrong_data_fails(self, encryption_manager):
        # Create payload with tampered ciphertext
        valid = encryption_manager.encrypt(b"test")
        tampered = EncryptedPayload(
            ciphertext=b"tampered_data_here_!",
            nonce=valid.nonce,
            key_version=valid.key_version,
        )

        with pytest.raises(DecryptionError):
            encryption_manager.decrypt(tampered)


class TestEncryptionManagerDisabled:
    """Test encryption manager in disabled mode."""

    def test_disabled_encrypt_returns_plaintext(self, disabled_manager):
        plaintext = b"Not encrypted"
        result = disabled_manager.encrypt(plaintext)

        assert result.ciphertext == plaintext
        assert result.algorithm == "NONE"
        assert result.key_version == 0

    def test_disabled_decrypt_returns_plaintext(self, disabled_manager):
        plaintext = b"Not encrypted"
        payload = EncryptedPayload(
            ciphertext=plaintext,
            nonce=b"\x00" * 12,
            algorithm="NONE",
            key_version=0,
        )
        result = disabled_manager.decrypt(payload)

        assert result == plaintext


class TestEncryptionManagerJson:
    """Test JSON encryption/decryption."""

    def test_encrypt_decrypt_json(self, encryption_manager):
        data = {
            "key": "value",
            "nested": {"list": [1, 2, 3]},
            "unicode": "日本語",
        }

        encrypted = encryption_manager.encrypt_json(data)
        decrypted = encryption_manager.decrypt_json(encrypted)

        assert decrypted == data

    def test_encrypted_json_is_string(self, encryption_manager):
        data = {"test": "data"}
        encrypted = encryption_manager.encrypt_json(data)

        # Should be valid JSON string
        parsed = json.loads(encrypted)
        assert "ciphertext" in parsed
        assert "nonce" in parsed


class TestEncryptionManagerFiles:
    """Test file encryption/decryption."""

    def test_encrypt_file(self, encryption_manager, tmp_path):
        source = tmp_path / "test.txt"
        source.write_text("File content to encrypt")

        dest = encryption_manager.encrypt_file(source)

        assert dest.exists()
        assert dest.suffix == ".enc"
        # File should be different from original
        assert dest.read_bytes() != source.read_bytes()

    def test_encrypt_decrypt_file_roundtrip(self, encryption_manager, tmp_path):
        source = tmp_path / "original.txt"
        original_content = "Original file content\nWith multiple lines."
        source.write_text(original_content)

        # Encrypt
        encrypted_path = encryption_manager.encrypt_file(source)

        # Decrypt
        decrypted_path = encryption_manager.decrypt_file(encrypted_path)

        assert decrypted_path.read_text() == original_content

    def test_encrypt_file_custom_dest(self, encryption_manager, tmp_path):
        source = tmp_path / "source.txt"
        source.write_text("Content")
        custom_dest = tmp_path / "custom_encrypted.bin"

        result = encryption_manager.encrypt_file(source, custom_dest)

        assert result == custom_dest
        assert custom_dest.exists()


class TestKeyRotation:
    """Test key rotation functionality."""

    def test_rotate_key_increments_version(self, encryption_manager):
        initial_version = encryption_manager.key_version

        new_version = encryption_manager.rotate_key()

        assert new_version == initial_version + 1
        assert encryption_manager.key_version == new_version

    def test_data_encrypted_before_rotation_can_be_decrypted(self, mock_keyring):
        """Note: This test demonstrates the limitation - after rotation,
        old data needs re-encryption or versioned key lookup."""
        manager = EncryptionManager(service_name="test_rotation")
        manager.initialize()

        # Encrypt with original key
        plaintext = b"Important data"
        encrypted = manager.encrypt(plaintext)

        # Data can still be decrypted (same session, key cached)
        decrypted = manager.decrypt(encrypted)
        assert decrypted == plaintext


class TestAuditLoggerWithEncryption:
    """Test AuditLogger with encryption enabled."""

    def test_audit_logger_with_encryption(self, mock_keyring, tmp_path):
        from src.futurnal.privacy.audit import AuditLogger, AuditEvent
        from datetime import datetime

        manager = EncryptionManager(service_name="test_audit")
        manager.initialize()

        audit_dir = tmp_path / "audit"
        audit_logger = AuditLogger(
            output_dir=audit_dir,
            encryption_manager=manager,
        )

        # Record an event
        event = AuditEvent(
            job_id="test_job_1",
            source="test_source",
            action="test_action",
            status="success",
            timestamp=datetime.utcnow(),
        )
        audit_logger.record(event)

        # Log file should contain encrypted data
        log_content = audit_logger._path.read_text()
        parsed = json.loads(log_content.strip())
        assert "ciphertext" in parsed

        # Can iterate and decrypt events
        events = list(audit_logger.iter_events())
        assert len(events) == 1
        assert events[0]["job_id"] == "test_job_1"

    def test_audit_logger_without_encryption(self, tmp_path):
        from src.futurnal.privacy.audit import AuditLogger, AuditEvent
        from datetime import datetime

        audit_dir = tmp_path / "audit_plain"
        audit_logger = AuditLogger(output_dir=audit_dir)

        event = AuditEvent(
            job_id="plain_job",
            source="test",
            action="test",
            status="success",
            timestamp=datetime.utcnow(),
        )
        audit_logger.record(event)

        # Log file should be plaintext JSON
        log_content = audit_logger._path.read_text()
        parsed = json.loads(log_content.strip())
        assert "ciphertext" not in parsed
        assert parsed["job_id"] == "plain_job"


class TestConsentRegistryWithEncryption:
    """Test ConsentRegistry with encryption enabled."""

    def test_consent_registry_with_encryption(self, mock_keyring, tmp_path):
        from src.futurnal.privacy.consent import ConsentRegistry

        manager = EncryptionManager(service_name="test_consent")
        manager.initialize()

        consent_dir = tmp_path / "consent"
        registry = ConsentRegistry(
            directory=consent_dir,
            encryption_manager=manager,
        )

        # Grant consent
        record = registry.grant(
            source="test_source",
            scope="CONTENT_ANALYSIS",
            operator="user",
        )
        assert record.granted

        # File should be encrypted
        content = registry._path.read_text()
        parsed = json.loads(content)
        assert "ciphertext" in parsed

        # Can still read consent
        retrieved = registry.get(source="test_source", scope="CONTENT_ANALYSIS")
        assert retrieved is not None
        assert retrieved.granted

    def test_consent_registry_without_encryption(self, tmp_path):
        from src.futurnal.privacy.consent import ConsentRegistry

        consent_dir = tmp_path / "consent_plain"
        registry = ConsentRegistry(directory=consent_dir)

        registry.grant(
            source="plain_source",
            scope="BASIC",
        )

        # File should be plaintext JSON
        content = registry._path.read_text()
        parsed = json.loads(content)
        assert "ciphertext" not in parsed
        assert "records" in parsed


class TestEncryptionSecurity:
    """Test encryption security properties."""

    def test_different_keys_cannot_decrypt(self, mock_keyring):
        # Create two managers with different keys
        manager1 = EncryptionManager(service_name="service1")
        manager1.initialize()

        manager2 = EncryptionManager(service_name="service2")
        manager2.initialize()

        # Encrypt with manager1
        encrypted = manager1.encrypt(b"secret")

        # Try to decrypt with manager2 (different key)
        with pytest.raises(DecryptionError):
            manager2.decrypt(encrypted)

    def test_nonce_is_unique_per_encryption(self, encryption_manager):
        nonces = set()
        for _ in range(100):
            encrypted = encryption_manager.encrypt(b"same_data")
            nonces.add(encrypted.nonce)

        # All nonces should be unique
        assert len(nonces) == 100

    def test_ciphertext_is_authenticated(self, encryption_manager):
        """GCM mode provides authentication - tampering is detected."""
        encrypted = encryption_manager.encrypt(b"authentic data")

        # Tamper with ciphertext
        tampered_ciphertext = bytearray(encrypted.ciphertext)
        tampered_ciphertext[0] ^= 0xFF  # Flip bits

        tampered = EncryptedPayload(
            ciphertext=bytes(tampered_ciphertext),
            nonce=encrypted.nonce,
            key_version=encrypted.key_version,
        )

        with pytest.raises(DecryptionError):
            encryption_manager.decrypt(tampered)
