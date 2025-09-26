"""Path redaction utilities for privacy-preserving telemetry and audit logs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RedactedPath:
    """Container for a redacted filesystem path and associated hashes."""

    redacted: str
    path_hash: str
    segments: tuple[str, ...]


@dataclass
class RedactionPolicy:
    """Configurable policy for masking filesystem paths."""

    reveal_filename: bool = True
    reveal_extension: bool = True
    segment_hash_length: int = 12
    path_hash_length: int = 40
    allow_plaintext: bool = False

    def apply(self, path: Path | str) -> RedactedPath:
        if self.allow_plaintext:
            resolved = str(path)
            return RedactedPath(
                redacted=resolved,
                path_hash=_hash_value(resolved, self.path_hash_length),
                segments=tuple(Path(resolved).parts),
            )

        text_path = str(path)
        resolved = Path(text_path)
        parts_list = list(resolved.parts)
        redacted_segments: list[str] = []
        for index, segment in enumerate(parts_list):
            is_last = index == len(parts_list) - 1
            if is_last and self.reveal_filename:
                redacted_segments.append(_redact_filename(segment, self.reveal_extension, self.segment_hash_length))
            else:
                redacted_segments.append(_hash_value(segment, self.segment_hash_length))

        redacted = "/".join(redacted_segments)
        return RedactedPath(
            redacted=redacted,
            path_hash=_hash_value(text_path, self.path_hash_length),
            segments=tuple(redacted_segments),
        )


_DEFAULT_POLICY = RedactionPolicy()


def redact_path(path: Path | str, *, policy: Optional[RedactionPolicy] = None) -> RedactedPath:
    """Return a policy-compliant redaction of ``path``.

    Parameters
    ----------
    path:
        Filesystem path to redact.
    policy:
        Optional custom policy. Defaults to global privacy policy.
    """

    active_policy = policy or _DEFAULT_POLICY
    return active_policy.apply(path)


def build_policy(*, allow_plaintext: bool = False) -> RedactionPolicy:
    """Construct a redaction policy with optional plaintext allowance."""

    if allow_plaintext:
        return RedactionPolicy(allow_plaintext=True)
    return _DEFAULT_POLICY


def _redact_filename(filename: str, reveal_extension: bool, hash_length: int) -> str:
    stem, suffix = _split_filename(filename)
    hashed = _hash_value(stem, hash_length)
    if reveal_extension and suffix:
        return f"file-{hashed}{suffix}"
    return f"file-{hashed}"


def _split_filename(filename: str) -> tuple[str, str]:
    if not filename:
        return "", ""
    path = Path(filename)
    suffix = "".join(path.suffixes)
    if suffix:
        stem = filename[: -len(suffix)]
    else:
        stem = filename
    return stem, suffix


def _hash_value(value: str, length: int) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest[:length]
