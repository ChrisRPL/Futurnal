"""Security and privacy audit tests for Futurnal.

These tests validate production readiness security requirements:
- Privacy compliance (consent, audit logging)
- Credential safety (no leaks in logs/telemetry)
- Consent flow integrity
- Audit log verification

Research Foundation:
- 2501.13904v3: Privacy-Preserving Personalized Federated Prompt Learning

Run with: pytest tests/security/ -v
"""
