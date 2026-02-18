# Privacy

Futurnal is built local-first.

## Principles

- Data stays on your machine by default.
- Access is consent-driven per data source.
- Cloud behavior is opt-in.
- You can revoke consent and delete data.

## What Futurnal Processes

When enabled by you, Futurnal can process:

- Obsidian vault content
- IMAP email content + metadata
- GitHub repositories/issues/PRs
- Local files

## What Futurnal Does Not Do By Default

- No default cloud upload of your source content
- No ad tracking
- No training remote models on your personal corpus

## Local Storage

Default runtime location:

- macOS/Linux: `~/.futurnal/`
- Windows: `%USERPROFILE%\\.futurnal\\`

## Credentials

Credentials should be stored with OS-provided secure storage (Keychain, Credential Manager, libsecret).

## Auditability

Futurnal includes privacy and audit commands for visibility into operations.
