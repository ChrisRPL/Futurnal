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
from .credential_manager import (
    AppPassword,
    CredentialManager,
    CredentialType,
    ImapCredential,
    OAuth2Tokens,
    OAuthProvider,
    OAuthProviderRegistry,
    auto_refresh_wrapper,
    clear_sensitive_string,
    secure_credential_context,
)
from .connection_manager import (
    ConnectionMetrics,
    ConnectionState,
    IdleConnection,
    ImapConnection,
    ImapConnectionPool,
    NetworkMonitor,
    RetryStrategy,
)
from .email_parser import (
    AttachmentMetadata,
    EmailAddress,
    EmailMessage,
    EmailParser,
)
from .email_normalizer import EmailNormalizer
from .email_triples import extract_email_triples
from .thread_models import (
    EmailThread,
    ParticipantRole,
    ThreadNode,
    ThreadParticipant,
)
from .thread_reconstructor import ThreadReconstructor
from .subject_evolution import SubjectEvolutionTracker
from .thread_triples import extract_thread_triples

__all__ = [
    "AuthMode",
    "ConsentScope",
    "ImapMailboxDescriptor",
    "ImapMailboxConfig",
    "MailboxPrivacySettings",
    "MailboxRegistry",
    "PrivacyLevel",
    "AppPassword",
    "CredentialManager",
    "CredentialType",
    "ImapCredential",
    "OAuth2Tokens",
    "OAuthProvider",
    "OAuthProviderRegistry",
    "auto_refresh_wrapper",
    "clear_sensitive_string",
    "secure_credential_context",
    "ConnectionMetrics",
    "ConnectionState",
    "IdleConnection",
    "ImapConnection",
    "ImapConnectionPool",
    "NetworkMonitor",
    "RetryStrategy",
    "AttachmentMetadata",
    "EmailAddress",
    "EmailMessage",
    "EmailParser",
    "EmailNormalizer",
    "extract_email_triples",
    "EmailThread",
    "ParticipantRole",
    "ThreadNode",
    "ThreadParticipant",
    "ThreadReconstructor",
    "SubjectEvolutionTracker",
    "extract_thread_triples",
]


