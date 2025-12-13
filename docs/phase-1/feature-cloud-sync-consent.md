Summary: Defines the cloud sync consent feature for optional PKG metadata backup to Firebase with explicit user consent.

# Feature - Cloud Sync Consent Flow

## Goal
Enable optional backup and synchronization of Personal Knowledge Graph (PKG) metadata to Firebase, providing users with explicit consent controls, automatic data deletion on revocation, and full audit transparency.

## Success Criteria
- Users presented with clear consent modal explaining exactly what metadata is synced (NOT document content).
- Consent stored permanently until explicitly revoked via ConsentRegistry.
- Automatic deletion of all cloud data when consent is revoked (privacy-first approach).
- Sync operates on periodic (15 min) + manual trigger basis while app is open.
- Audit logs capture all sync operations and consent decisions.
- Flow integrates seamlessly into desktop shell Privacy Settings.

## Functional Scope
- Consent prompt UI with clear explanation of data categories (graph structure, settings, search history).
- Consent storage integrated with existing ConsentRegistry (encrypted, single source of truth).
- Firestore sync client with consent gate checks before any write operation.
- Revocation management with immediate propagation and automatic cloud data deletion.
- Periodic sync (every 15 minutes) with manual "Sync Now" option.
- Sync on app close if changes pending.
- Telemetry capturing sync frequency, success/failure rates, and user consent decisions.

## Non-Functional Guarantees
- Metadata-only sync: NO document content, email bodies, or file contents leave the device.
- Consent duration: Permanent until explicitly revoked (no renewal friction).
- Revocation: Automatic deletion of ALL user data from Firebase (clean break).
- Optional client-side encryption before Firebase upload.
- No background sync when app is closed (privacy + battery concerns).
- Performance overhead under 5% for sync operations.

## Dependencies
- Privacy & audit logging feature ([feature-privacy-audit-logging](feature-privacy-audit-logging.md)).
- Firebase SDK already initialized in desktop app (authentication working).
- Tauri IPC layer for consent state communication.
- ConsentRegistry and AuditLogger in Python backend.

## Implementation Guide
1. **Consent Scope Definition:** Define CloudSyncConsentScopes enum with PKG_METADATA_BACKUP, PKG_SETTINGS_BACKUP, SEARCH_HISTORY_SYNC.
2. **CloudSyncConsentManager:** Implement consent enforcement class coordinating with ConsentRegistry and AuditLogger.
3. **CLI Commands:** Add `futurnal cloud-sync consent status/grant/revoke` and `futurnal cloud-sync audit` commands.
4. **IPC Layer:** Implement Rust commands in cloud_sync.rs following existing privacy.rs patterns.
5. **Consent Modal UI:** Create multi-step wizard with clear data category explanations and confirmation step.
6. **Zustand Store:** Implement cloudSyncStore managing consent state, sync status, and periodic sync logic.
7. **Firebase Client:** Build cloudSync.ts with consent-gated Firestore operations, offline queue, and auto-delete on revocation.
8. **Privacy Section Update:** Add Cloud Backup section to settings with consent status, sync toggle, and revoke button.

## Testing Strategy
- **Unit Tests:** Consent scope validation, grant/revoke logic, metadata export structure.
- **Integration Tests:** Full consent flow from UI through IPC to Python ConsentRegistry.
- **E2E Tests:** User grants consent, data syncs to Firestore, user revokes, data is deleted.
- **Security Tests:** Attempt sync without consent, verify blocking.
- **Performance Tests:** Measure sync overhead, ensure under 5%.

## Code Review Checklist
- Consent prompts use clear, non-technical language.
- All Firestore write operations gated by consent check.
- Revocation triggers immediate sync stop AND automatic cloud data deletion.
- Audit logging captures sync operations without exposing content.
- No document content, email bodies, or file contents included in sync payload.
- TypeScript and Python types are consistent.
- Tests cover consent grant, revoke, sync, and edge cases.

## Documentation & Follow-up
- Update Privacy Settings documentation with cloud sync explanation.
- Add FAQ entry: "What data is synced to the cloud?"
- Coordinate with future cross-device sync features.

## Out of Scope
- Cloud LLM escalation (covered by separate feature-cloud-escalation-consent.md).
- Real-time collaborative editing.
- Full document content sync.
- Background sync when app is closed.

## Consent Scopes

| Scope | Description | Required |
|-------|-------------|----------|
| `cloud:pkg:metadata_backup` | Graph node IDs, types, labels, timestamps, relationship types | Required for any sync |
| `cloud:pkg:settings_backup` | User preferences and app settings | Optional |
| `cloud:search:history_sync` | Search query history | Optional (disabled by default) |
