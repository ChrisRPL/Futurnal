"""Compatibility shim for legacy imports of AuditLogger."""

from __future__ import annotations

from ..privacy.audit import AuditEvent, AuditLogger

__all__ = ["AuditEvent", "AuditLogger"]


