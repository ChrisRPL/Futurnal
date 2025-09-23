Summary: Captures the cloud escalation consent flow feature with scope, testing, and review checkpoints.

# Feature · Cloud Escalation Consent Flow

## Goal
Provide a structured flow for requesting, recording, and enforcing user consent when Futurnal escalates tasks to cloud-based models or services beyond the on-device defaults.

## Success Criteria
- Users presented with clear explanation of data shared, model used, and purpose before escalation.
- Consent stored with scope (action, data type), duration, and revocation capability.
- Requests blocked automatically if consent missing or expired.
- Audit logs capture escalation requests and outcomes per privacy requirements.
- Flow integrates seamlessly into desktop shell and CLIs.

## Functional Scope
- Consent prompt UI with explanations, privacy policy links, and option to remember decisions.
- Consent storage integrated with privacy registry and keychain.
- Enforcement middleware gating cloud API calls.
- Revocation management with immediate effect across system components.
- Telemetry capturing escalation frequency and user choices.

## Non-Functional Guarantees
- Clear, non-technical language for consent copy; accessible design.
- Secure storage of consent records; encrypted at rest.
- Offline fallback: queue requests until consent granted or user returns online.
- Logging anonymized to protect sensitive context.

## Dependencies
- Privacy & audit logging feature ([feature-privacy-audit-logging](feature-privacy-audit-logging.md)).
- Components capable of cloud escalation (entity extraction, search summarization).
- Desktop shell for user interaction surfaces.

## Implementation Guide
1. **Consent Policy Definition:** Specify scopes (e.g., "Summarize via GPT-4o"), retention, and default states per [requirements/system-requirements.md](../requirements/system-requirements.md).
2. **UX Design:** Collaborate with design to craft accessible modal and CLI prompts; reference product vision for tone.
3. **Storage & Enforcement:** Implement consent registry shared with privacy feature; enforce checks at API client layer using automata-based state machine for request lifecycle.
4. **Revocation Flow:** Provide UI/CLI to revoke or modify consent; propagate changes immediately.
5. **Telemetry & Audit:** Log every escalation attempt, consent status, and resulting action.
6. **Education Hooks:** Link to docs explaining on-device vs. cloud trade-offs; allow per-task override.

## Testing Strategy
- **Unit Tests:** Consent state transitions, storage encryption, expiration handling.
- **Integration Tests:** Simulate escalations from various components; verify enforcement and logging.
- **UX Tests:** Accessibility evaluation, clarity of messaging (user testing).
- **Security Tests:** Attempt escalations without consent, verify blocking.

## Code Review Checklist
- Consent prompts accurate, localized (if applicable), and accessible.
- Enforcement middleware covers all cloud-bound pathways.
- Revocation updates propagate without caching delays.
- Telemetry and audit logging capture necessary context without exposing data.
- Documentation updated with user guidance on consent management.

## Documentation & Follow-up
- Update user privacy docs with cloud escalation explanation.
- Train support team on consent troubleshooting.
- Coordinate with future Analyst/Guide features requiring richer escalations.


