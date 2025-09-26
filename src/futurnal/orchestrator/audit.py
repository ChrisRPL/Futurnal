"""Compatibility shim for legacy imports of audit primitives."""

from __future__ import annotations

from ..privacy.audit import AuditEvent, AuditLogger


__all__ = ["AuditEvent", "AuditLogger"]


