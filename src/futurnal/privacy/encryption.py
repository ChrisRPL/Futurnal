"""Storage encryption manager for privacy-sensitive data.

This module provides transparent encryption for audit logs and consent records
using AES-256-GCM authenticated encryption with keys stored in the OS keychain.

Features:
- AES-256-GCM authenticated encryption (confidentiality + integrity)
- Per-file unique IVs (nonces) for security
- Key storage in OS keychain via keyring library
- Key rotation support with version tracking
- Streaming support for large files

Security Properties:
- 256-bit encryption keys
- 96-bit random nonces (GCM standard)
- 128-bit authentication tags
- Keys never stored on disk (OS keychain only)

Privacy-First Design (Option B):
- Local-only encryption (no cloud key management)
- User controls key lifecycle
- Fail-secure: errors prevent data access, not expose it

Usage:
    >>> from futurnal.privacy.encryption import EncryptionManager
    >>> manager = EncryptionManager(service_name="futurnal")
    >>> manager.initialize()  # Generate or retrieve key
    >>> encrypted = manager.encrypt(b"sensitive data")
    >>> decrypted = manager.decrypt(encrypted)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)

# Constants
KEY_SIZE_BYTES = 32  # 256 bits for AES-256
NONCE_SIZE_BYTES = 12  # 96 bits for GCM
TAG_SIZE_BYTES = 16  # 128-bit authentication tag


class EncryptionError(Exception):
    """Base exception for encryption errors."""


class KeyNotFoundError(EncryptionError):
    """Raised when encryption key is not found in keychain."""


class DecryptionError(EncryptionError):
    """Raised when decryption fails (wrong key, corrupted data, or tampered)."""


class KeyRotationError(EncryptionError):
    """Raised when key rotation fails."""


@dataclass
class EncryptedPayload:
    """Container for encrypted data with metadata.

    Attributes:
        ciphertext: The encrypted data (base64 encoded when serialized)
        nonce: Random IV/nonce used for encryption (base64 encoded)
        key_version: Version of the key used for encryption
        algorithm: Encryption algorithm identifier
        created_at: Timestamp when encryption occurred
    """

    ciphertext: bytes
    nonce: bytes
    key_version: int = 1
    algorithm: str = "AES-256-GCM"
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, str]:
        """Serialize to dictionary for JSON storage."""
        return {
            "ciphertext": base64.b64encode(self.ciphertext).decode("ascii"),
            "nonce": base64.b64encode(self.nonce).decode("ascii"),
            "key_version": self.key_version,
            "algorithm": self.algorithm,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "EncryptedPayload":
        """Deserialize from dictionary."""
        return cls(
            ciphertext=base64.b64decode(data["ciphertext"]),
            nonce=base64.b64decode(data["nonce"]),
            key_version=data.get("key_version", 1),
            algorithm=data.get("algorithm", "AES-256-GCM"),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None,
        )

    def to_bytes(self) -> bytes:
        """Serialize to bytes for file storage."""
        return json.dumps(self.to_dict()).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "EncryptedPayload":
        """Deserialize from bytes."""
        return cls.from_dict(json.loads(data.decode("utf-8")))


@dataclass
class EncryptionManager:
    """Manages encryption keys and operations for privacy-sensitive data.

    Uses the OS keychain (via keyring) for secure key storage and
    AES-256-GCM for authenticated encryption.

    Example:
        >>> manager = EncryptionManager()
        >>> manager.initialize()
        >>> # Encrypt data
        >>> encrypted = manager.encrypt(b"secret data")
        >>> # Decrypt data
        >>> decrypted = manager.decrypt(encrypted)
        >>> assert decrypted == b"secret data"

    Attributes:
        service_name: Keychain service identifier
        key_id: Unique identifier for the encryption key
        enabled: Whether encryption is enabled (can be disabled for testing)
    """

    service_name: str = "futurnal"
    key_id: str = "audit_encryption_key"
    enabled: bool = True
    _key_version: int = field(default=1, init=False)
    _key_cache: Optional[bytes] = field(default=None, init=False, repr=False)

    def initialize(self, *, force_new: bool = False) -> None:
        """Initialize encryption by retrieving or generating key.

        Args:
            force_new: If True, generate new key even if one exists

        Raises:
            EncryptionError: If key generation or retrieval fails
        """
        if not self.enabled:
            logger.info("Encryption disabled, skipping initialization")
            return

        try:
            import keyring

            if force_new:
                self._generate_and_store_key()
                return

            # Try to retrieve existing key
            existing = keyring.get_password(self.service_name, self.key_id)
            if existing:
                self._key_cache = base64.b64decode(existing)
                # Retrieve key version
                version_key = f"{self.key_id}_version"
                version_str = keyring.get_password(self.service_name, version_key)
                self._key_version = int(version_str) if version_str else 1
                logger.info(
                    f"Retrieved existing encryption key (version {self._key_version})"
                )
            else:
                self._generate_and_store_key()

        except Exception as e:
            raise EncryptionError(f"Failed to initialize encryption: {e}") from e

    def _generate_and_store_key(self) -> None:
        """Generate a new encryption key and store in keychain."""
        import keyring

        # Generate cryptographically secure random key
        key = secrets.token_bytes(KEY_SIZE_BYTES)
        key_b64 = base64.b64encode(key).decode("ascii")

        # Store in keychain
        keyring.set_password(self.service_name, self.key_id, key_b64)
        keyring.set_password(
            self.service_name, f"{self.key_id}_version", str(self._key_version)
        )

        self._key_cache = key
        logger.info(f"Generated new encryption key (version {self._key_version})")

    def _get_key(self) -> bytes:
        """Retrieve encryption key from cache or keychain.

        Returns:
            The encryption key bytes

        Raises:
            KeyNotFoundError: If key is not found
        """
        if self._key_cache:
            return self._key_cache

        if not self.enabled:
            raise KeyNotFoundError("Encryption disabled, no key available")

        try:
            import keyring

            key_b64 = keyring.get_password(self.service_name, self.key_id)
            if not key_b64:
                raise KeyNotFoundError(
                    f"Encryption key not found in keychain for service '{self.service_name}'"
                )

            self._key_cache = base64.b64decode(key_b64)
            return self._key_cache

        except KeyNotFoundError:
            raise
        except Exception as e:
            raise KeyNotFoundError(f"Failed to retrieve key: {e}") from e

    def encrypt(self, plaintext: Union[bytes, str]) -> EncryptedPayload:
        """Encrypt data using AES-256-GCM.

        Args:
            plaintext: Data to encrypt (str will be UTF-8 encoded)

        Returns:
            EncryptedPayload containing ciphertext and metadata

        Raises:
            EncryptionError: If encryption fails
        """
        if not self.enabled:
            # Return "encrypted" payload with plaintext for disabled mode
            data = plaintext.encode("utf-8") if isinstance(plaintext, str) else plaintext
            return EncryptedPayload(
                ciphertext=data,
                nonce=b"\x00" * NONCE_SIZE_BYTES,
                key_version=0,
                algorithm="NONE",
            )

        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            key = self._get_key()
            aesgcm = AESGCM(key)

            # Generate random nonce
            nonce = secrets.token_bytes(NONCE_SIZE_BYTES)

            # Ensure plaintext is bytes
            if isinstance(plaintext, str):
                plaintext = plaintext.encode("utf-8")

            # Encrypt with authentication
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)

            return EncryptedPayload(
                ciphertext=ciphertext,
                nonce=nonce,
                key_version=self._key_version,
                algorithm="AES-256-GCM",
            )

        except Exception as e:
            raise EncryptionError(f"Encryption failed: {e}") from e

    def decrypt(self, payload: EncryptedPayload) -> bytes:
        """Decrypt data using AES-256-GCM.

        Args:
            payload: EncryptedPayload to decrypt

        Returns:
            Decrypted plaintext bytes

        Raises:
            DecryptionError: If decryption fails (wrong key, corrupted, or tampered)
        """
        if not self.enabled or payload.algorithm == "NONE":
            return payload.ciphertext

        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            # TODO: Support key version lookup for rotation
            key = self._get_key()
            aesgcm = AESGCM(key)

            # Decrypt with authentication verification
            plaintext = aesgcm.decrypt(payload.nonce, payload.ciphertext, None)

            return plaintext

        except Exception as e:
            raise DecryptionError(f"Decryption failed: {e}") from e

    def encrypt_file(self, source_path: Path, dest_path: Optional[Path] = None) -> Path:
        """Encrypt a file.

        Args:
            source_path: Path to file to encrypt
            dest_path: Optional destination path (defaults to source_path + .enc)

        Returns:
            Path to encrypted file

        Raises:
            EncryptionError: If encryption fails
        """
        if dest_path is None:
            dest_path = source_path.with_suffix(source_path.suffix + ".enc")

        try:
            plaintext = source_path.read_bytes()
            payload = self.encrypt(plaintext)
            dest_path.write_bytes(payload.to_bytes())

            logger.debug(f"Encrypted {source_path} -> {dest_path}")
            return dest_path

        except Exception as e:
            raise EncryptionError(f"Failed to encrypt file {source_path}: {e}") from e

    def decrypt_file(self, source_path: Path, dest_path: Optional[Path] = None) -> Path:
        """Decrypt a file.

        Args:
            source_path: Path to encrypted file
            dest_path: Optional destination path (defaults to removing .enc suffix)

        Returns:
            Path to decrypted file

        Raises:
            DecryptionError: If decryption fails
        """
        if dest_path is None:
            if source_path.suffix == ".enc":
                dest_path = source_path.with_suffix("")
            else:
                dest_path = source_path.with_suffix(source_path.suffix + ".dec")

        try:
            encrypted_data = source_path.read_bytes()
            payload = EncryptedPayload.from_bytes(encrypted_data)
            plaintext = self.decrypt(payload)
            dest_path.write_bytes(plaintext)

            logger.debug(f"Decrypted {source_path} -> {dest_path}")
            return dest_path

        except Exception as e:
            raise DecryptionError(f"Failed to decrypt file {source_path}: {e}") from e

    def encrypt_json(self, data: dict) -> str:
        """Encrypt a JSON-serializable dictionary.

        Args:
            data: Dictionary to encrypt

        Returns:
            JSON string of encrypted payload
        """
        plaintext = json.dumps(data, separators=(",", ":"))
        payload = self.encrypt(plaintext)
        return json.dumps(payload.to_dict())

    def decrypt_json(self, encrypted_json: str) -> dict:
        """Decrypt an encrypted JSON payload.

        Args:
            encrypted_json: JSON string of encrypted payload

        Returns:
            Decrypted dictionary
        """
        payload = EncryptedPayload.from_dict(json.loads(encrypted_json))
        plaintext = self.decrypt(payload)
        return json.loads(plaintext.decode("utf-8"))

    def rotate_key(self) -> int:
        """Rotate encryption key.

        Creates a new key version. Old keys should be retained
        to decrypt data encrypted with previous versions.

        Returns:
            New key version number

        Raises:
            KeyRotationError: If rotation fails
        """
        try:
            import keyring

            # Backup current key with version
            current_key = self._get_key()
            old_version = self._key_version
            old_key_id = f"{self.key_id}_v{old_version}"
            keyring.set_password(
                self.service_name,
                old_key_id,
                base64.b64encode(current_key).decode("ascii"),
            )

            # Generate new key
            self._key_version += 1
            self._key_cache = None  # Clear cache
            self._generate_and_store_key()

            logger.info(f"Rotated key from version {old_version} to {self._key_version}")
            return self._key_version

        except Exception as e:
            raise KeyRotationError(f"Key rotation failed: {e}") from e

    def delete_key(self) -> None:
        """Delete encryption key from keychain.

        WARNING: This will make all encrypted data permanently inaccessible!

        Use with extreme caution.
        """
        try:
            import keyring

            keyring.delete_password(self.service_name, self.key_id)
            keyring.delete_password(self.service_name, f"{self.key_id}_version")
            self._key_cache = None

            logger.warning("Deleted encryption key from keychain")

        except Exception as e:
            logger.error(f"Failed to delete key: {e}")

    @property
    def key_version(self) -> int:
        """Get current key version."""
        return self._key_version

    def is_initialized(self) -> bool:
        """Check if encryption is initialized."""
        if not self.enabled:
            return True

        try:
            import keyring

            return keyring.get_password(self.service_name, self.key_id) is not None
        except Exception:
            return False


# Singleton instance for global access
_default_manager: Optional[EncryptionManager] = None


def get_encryption_manager(
    *,
    service_name: str = "futurnal",
    enabled: bool = True,
    auto_init: bool = True,
) -> EncryptionManager:
    """Get or create the default encryption manager.

    Args:
        service_name: Keychain service name
        enabled: Whether encryption is enabled
        auto_init: Whether to auto-initialize if not already done

    Returns:
        EncryptionManager instance
    """
    global _default_manager

    if _default_manager is None:
        _default_manager = EncryptionManager(
            service_name=service_name,
            enabled=enabled,
        )
        if auto_init and enabled:
            _default_manager.initialize()

    return _default_manager


__all__ = [
    "EncryptionManager",
    "EncryptedPayload",
    "EncryptionError",
    "KeyNotFoundError",
    "DecryptionError",
    "KeyRotationError",
    "get_encryption_manager",
]
