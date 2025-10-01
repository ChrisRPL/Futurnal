"""Privacy policy implementation for Obsidian vault operations.

This module provides a comprehensive privacy policy system that governs data handling,
consent management, and audit requirements for Obsidian vault processing.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Set, Dict, Any

from pydantic import BaseModel, Field

from ...privacy.consent import ConsentRegistry, ConsentRecord, ConsentRequiredError
from ...privacy.redaction import RedactionPolicy, RedactedPath
from .descriptor import VaultPrivacySettings, PrivacyLevel, ConsentScope


class ObsidianPrivacyPolicy:
    """Comprehensive privacy policy for Obsidian vault operations.

    This class encapsulates all privacy-related decisions and provides a unified
    interface for consent checks, redaction policies, and audit requirements.
    """

    def __init__(
        self,
        vault_id: str,
        privacy_settings: VaultPrivacySettings,
        title_patterns: Optional[List[str]] = None,
        consent_registry: Optional[ConsentRegistry] = None,
    ):
        self.vault_id = vault_id
        self.privacy_settings = privacy_settings
        self.title_patterns = [re.compile(p) for p in (title_patterns or [])]
        self.consent_registry = consent_registry

    def requires_consent_for_operation(self, operation: ConsentScope) -> bool:
        """Check if an operation requires consent for this vault."""
        return operation in self.privacy_settings.required_consent_scopes

    def check_consent(self, operation: ConsentScope, *, operator: Optional[str] = None) -> bool:
        """Check if consent is granted for a specific operation.

        Args:
            operation: The operation being requested
            operator: Optional operator identifier

        Returns:
            True if consent is granted or not required, False otherwise

        Raises:
            ConsentRequiredError: If consent is required but not granted
        """
        if not self.requires_consent_for_operation(operation):
            return True

        if self.consent_registry is None:
            if operation in self.privacy_settings.required_consent_scopes:
                raise ConsentRequiredError(f"Consent registry not available for {operation.value}")
            return True

        try:
            consent_record = self.consent_registry.require(
                source=f"obsidian_vault:{self.vault_id}",
                scope=operation.value
            )
            return consent_record.is_active()
        except ConsentRequiredError:
            return False

    def grant_consent(
        self,
        operation: ConsentScope,
        *,
        operator: Optional[str] = None,
        duration_hours: Optional[int] = None,
        token: Optional[str] = None,
    ) -> Optional[ConsentRecord]:
        """Grant consent for a specific operation.

        Args:
            operation: The operation to grant consent for
            operator: Optional operator identifier
            duration_hours: Optional expiration time in hours
            token: Optional authorization token

        Returns:
            ConsentRecord if consent was granted, None if no registry available
        """
        if self.consent_registry is None:
            return None

        return self.consent_registry.grant(
            source=f"obsidian_vault:{self.vault_id}",
            scope=operation.value,
            operator=operator,
            duration_hours=duration_hours,
            token=token,
        )

    def revoke_consent(
        self,
        operation: ConsentScope,
        *,
        operator: Optional[str] = None,
    ) -> Optional[ConsentRecord]:
        """Revoke consent for a specific operation.

        Args:
            operation: The operation to revoke consent for
            operator: Optional operator identifier

        Returns:
            ConsentRecord if consent was revoked, None if no registry available
        """
        if self.consent_registry is None:
            return None

        return self.consent_registry.revoke(
            source=f"obsidian_vault:{self.vault_id}",
            scope=operation.value,
            operator=operator,
        )

    def build_redaction_policy(self, *, override_plaintext: Optional[bool] = None) -> RedactionPolicy:
        """Build a redaction policy based on vault privacy settings.

        Args:
            override_plaintext: Optional override for plaintext allowance

        Returns:
            Configured RedactionPolicy instance
        """
        # Determine plaintext allowance
        allow_plaintext = override_plaintext
        if allow_plaintext is None:
            allow_plaintext = (
                self.privacy_settings.privacy_level == PrivacyLevel.PERMISSIVE and
                not self.privacy_settings.enable_path_anonymization
            )

        class ObsidianRedactionPolicy(RedactionPolicy):
            def __init__(self, policy_instance: ObsidianPrivacyPolicy, **kwargs):
                super().__init__(**kwargs)
                self.policy = policy_instance

            def apply(self, path: Path | str) -> RedactedPath:
                path_obj = Path(path)
                should_force_redaction = self._check_sensitive_content(path_obj)

                # Apply strict redaction for sensitive content
                if should_force_redaction or self.policy.privacy_settings.privacy_level == PrivacyLevel.STRICT:
                    temp_policy = RedactionPolicy(
                        reveal_filename=False,
                        reveal_extension=False if self.policy.privacy_settings.privacy_level == PrivacyLevel.STRICT else self.reveal_extension,
                        allow_plaintext=False,
                        segment_hash_length=self.segment_hash_length,
                        path_hash_length=self.path_hash_length,
                    )
                    return temp_policy.apply(path)

                # Use standard redaction
                return super().apply(path)

            def _check_sensitive_content(self, path: Path) -> bool:
                """Check if content should be treated as sensitive."""
                if path.suffix.lower() not in ['.md', '.markdown']:
                    return False

                stem = path.stem

                # Check title patterns
                for pattern in self.policy.title_patterns:
                    if pattern.search(stem):
                        return True

                # Check privacy tags if enabled
                if self.policy.privacy_settings.tag_based_privacy_classification:
                    for tag in self.policy.privacy_settings.privacy_tags:
                        if tag.lower() in stem.lower():
                            return True

                return False

        return ObsidianRedactionPolicy(
            policy_instance=self,
            allow_plaintext=allow_plaintext and self.privacy_settings.enable_path_anonymization,
            reveal_filename=self.privacy_settings.privacy_level != PrivacyLevel.STRICT,
            reveal_extension=self.privacy_settings.privacy_level == PrivacyLevel.PERMISSIVE,
        )

    def should_audit_event(self, event_type: str) -> bool:
        """Determine if an event type should be audited based on privacy settings.

        Args:
            event_type: Type of event (e.g., 'content_change', 'link_change')

        Returns:
            True if the event should be audited
        """
        if event_type == 'content_change':
            return self.privacy_settings.audit_content_changes
        elif event_type == 'link_change':
            return self.privacy_settings.audit_link_changes
        else:
            # Default to auditing for unknown event types (fail-safe)
            return True

    def get_audit_retention_days(self) -> int:
        """Get the audit log retention period for this vault."""
        return self.privacy_settings.retain_audit_days

    def is_content_redaction_enabled(self) -> bool:
        """Check if content redaction is enabled."""
        return self.privacy_settings.enable_content_redaction

    def is_path_anonymization_enabled(self) -> bool:
        """Check if path anonymization is enabled."""
        return self.privacy_settings.enable_path_anonymization

    def get_privacy_level(self) -> PrivacyLevel:
        """Get the privacy level for this vault."""
        return self.privacy_settings.privacy_level

    def get_required_consent_scopes(self) -> List[ConsentScope]:
        """Get list of required consent scopes."""
        return self.privacy_settings.required_consent_scopes

    def validate_operation_consent(self, operations: List[ConsentScope]) -> Dict[ConsentScope, bool]:
        """Validate consent for multiple operations.

        Args:
            operations: List of operations to check

        Returns:
            Dictionary mapping operations to consent status
        """
        results = {}
        for operation in operations:
            try:
                results[operation] = self.check_consent(operation)
            except ConsentRequiredError:
                results[operation] = False
        return results

    def get_all_consent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive consent status for all scopes.

        Returns:
            Dictionary with consent status information
        """
        status = {}
        for scope in ConsentScope:
            try:
                is_granted = self.check_consent(scope)
                consent_record = None

                if self.consent_registry:
                    try:
                        consent_record = self.consent_registry.get(
                            source=f"obsidian_vault:{self.vault_id}",
                            scope=scope.value
                        )
                    except:
                        pass

                status[scope.value] = {
                    "required": self.requires_consent_for_operation(scope),
                    "granted": is_granted,
                    "expires_at": consent_record.expires_at.isoformat() if consent_record and consent_record.expires_at else None,
                    "granted_at": consent_record.timestamp.isoformat() if consent_record else None,
                    "operator": consent_record.operator if consent_record else None,
                }
            except Exception:
                status[scope.value] = {
                    "required": self.requires_consent_for_operation(scope),
                    "granted": False,
                    "error": "Failed to check consent status"
                }

        return status

    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary representation."""
        return {
            "vault_id": self.vault_id,
            "privacy_settings": self.privacy_settings.model_dump(),
            "title_patterns": [p.pattern for p in self.title_patterns],
            "consent_status": self.get_all_consent_status(),
        }

    @classmethod
    def from_vault_descriptor(
        cls,
        vault_descriptor,  # ObsidianVaultDescriptor
        consent_registry: Optional[ConsentRegistry] = None,
    ) -> ObsidianPrivacyPolicy:
        """Create privacy policy from vault descriptor.

        Args:
            vault_descriptor: ObsidianVaultDescriptor instance
            consent_registry: Optional consent registry

        Returns:
            Configured ObsidianPrivacyPolicy instance
        """
        return cls(
            vault_id=vault_descriptor.id,
            privacy_settings=vault_descriptor.privacy_settings,
            title_patterns=vault_descriptor.redact_title_patterns,
            consent_registry=consent_registry,
        )


class VaultConsentManager:
    """Simplified consent management interface for vault operations."""

    def __init__(self, privacy_policy: ObsidianPrivacyPolicy):
        self.policy = privacy_policy

    def require_consent_for_scan(self) -> bool:
        """Check if consent is granted for basic vault scanning."""
        return self.policy.check_consent(ConsentScope.VAULT_SCAN)

    def require_consent_for_content_analysis(self) -> bool:
        """Check if consent is granted for content analysis."""
        return self.policy.check_consent(ConsentScope.CONTENT_ANALYSIS)

    def require_consent_for_asset_extraction(self) -> bool:
        """Check if consent is granted for asset extraction."""
        return self.policy.check_consent(ConsentScope.ASSET_EXTRACTION)

    def require_consent_for_link_analysis(self) -> bool:
        """Check if consent is granted for link graph analysis."""
        return self.policy.check_consent(ConsentScope.LINK_GRAPH_ANALYSIS)

    def require_consent_for_cloud_models(self) -> bool:
        """Check if consent is granted for cloud model usage."""
        return self.policy.check_consent(ConsentScope.CLOUD_MODELS)

    def require_consent_for_metadata_extraction(self) -> bool:
        """Check if consent is granted for metadata extraction."""
        return self.policy.check_consent(ConsentScope.METADATA_EXTRACTION)

    def grant_all_required_consent(
        self,
        *,
        operator: Optional[str] = None,
        duration_hours: Optional[int] = None,
    ) -> List[ConsentRecord]:
        """Grant consent for all required operations.

        Args:
            operator: Optional operator identifier
            duration_hours: Optional expiration time in hours

        Returns:
            List of granted consent records
        """
        records = []
        for scope in self.policy.get_required_consent_scopes():
            record = self.policy.grant_consent(
                scope,
                operator=operator,
                duration_hours=duration_hours,
            )
            if record:
                records.append(record)
        return records

    def revoke_all_consent(self, *, operator: Optional[str] = None) -> List[ConsentRecord]:
        """Revoke consent for all operations.

        Args:
            operator: Optional operator identifier

        Returns:
            List of revoked consent records
        """
        records = []
        for scope in ConsentScope:
            record = self.policy.revoke_consent(scope, operator=operator)
            if record:
                records.append(record)
        return records

    def get_missing_consent(self) -> List[ConsentScope]:
        """Get list of required but missing consent scopes."""
        missing = []
        for scope in self.policy.get_required_consent_scopes():
            try:
                if not self.policy.check_consent(scope):
                    missing.append(scope)
            except ConsentRequiredError:
                missing.append(scope)
        return missing