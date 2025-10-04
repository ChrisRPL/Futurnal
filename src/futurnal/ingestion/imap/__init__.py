"""IMAP mailbox descriptor and registry utilities."""

from .descriptor import (
    AuthMode,
    ConsentScope,
    ImapMailboxDescriptor,
    MailboxPrivacySettings,
    MailboxRegistry,
    PrivacyLevel,
)
from .config import ImapMailboxConfig

__all__ = [
    "AuthMode",
    "ConsentScope",
    "ImapMailboxDescriptor",
    "ImapMailboxConfig",
    "MailboxPrivacySettings",
    "MailboxRegistry",
    "PrivacyLevel",
]


