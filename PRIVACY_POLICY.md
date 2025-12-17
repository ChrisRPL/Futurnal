# Privacy Policy

**Futurnal** - Privacy-First Personal Knowledge Engine

*Last updated: December 2024*
*Version: 1.0.0*

## Our Commitment

Futurnal is built on a fundamental principle: **your data belongs to you**. We are committed to:

1. **Local-First Processing**: All data processing happens on your device
2. **Explicit Consent**: No data access without your clear permission
3. **Transparency**: Clear explanations of what we do with your data
4. **Control**: You can delete your data at any time

## What Data Futurnal Processes

### Data You Provide

When you connect data sources to Futurnal, we process:

| Source Type | Data Processed |
|-------------|----------------|
| Obsidian Vault | Markdown content, frontmatter, wikilinks |
| Email (IMAP) | Email subject, body, metadata, attachments |
| GitHub | Code, issues, PRs, wiki, commit messages |
| Local Files | File content based on supported types |

### How Data is Processed

1. **Extraction**: Content is parsed and normalized
2. **Embedding**: Text is converted to vector representations
3. **Graph Storage**: Relationships are stored in a local knowledge graph
4. **Index**: Content is indexed for search

**All processing occurs locally on your device.**

## What We Do NOT Collect

Futurnal does NOT:

- Send your data to external servers (by default)
- Track your searches or queries
- Store your credentials on remote servers
- Share your data with third parties
- Use your data for advertising
- Train AI models with your personal data

## Data Storage

### Local Storage

All Futurnal data is stored locally in:
- **macOS/Linux**: `~/.futurnal/`
- **Windows**: `%USERPROFILE%\.futurnal\`

This includes:
- Knowledge graph database
- Vector embeddings
- Configuration files
- Audit logs
- Session data

### Credential Storage

Sensitive credentials (email passwords, API tokens) are stored using your operating system's secure credential storage:
- **macOS**: Keychain
- **Windows**: Credential Manager
- **Linux**: libsecret/GNOME Keyring

## Consent System

### How Consent Works

Before accessing any data source, Futurnal requires explicit consent:

1. **Scope-Based**: Consent is granted per data source
2. **Granular**: You control what operations are allowed
3. **Revocable**: You can revoke consent at any time
4. **Logged**: All consent changes are audit-logged

### Consent Scopes

| Scope | Description |
|-------|-------------|
| `read` | Permission to read from the source |
| `process` | Permission to extract and process content |
| `store` | Permission to store in knowledge graph |

### Managing Consent

```bash
# View consent status
futurnal privacy consent list

# Revoke consent
futurnal privacy consent revoke --source "vault:my-vault"

# Revoke and delete data
futurnal privacy consent revoke --source "vault:my-vault" --delete-data
```

## Audit Logging

### What is Logged

Futurnal maintains tamper-evident audit logs of:
- Data source access events
- Consent changes
- Search operations (metadata only)
- Data modifications

### What is NOT Logged

- Query content (your actual searches)
- Document content
- Personal identifiers
- Credential values

### Log Format Example

```json
{
  "timestamp": "2024-12-17T10:30:00Z",
  "action": "search_executed",
  "metadata": {
    "search_type": "temporal",
    "result_count": 15,
    "latency_ms": 250
  }
}
```

### Verifying Log Integrity

```bash
futurnal privacy audit verify
```

## Telemetry (Optional)

### Disabled by Default

Telemetry is **opt-in only**. It is disabled by default.

### What Telemetry Collects (If Enabled)

- Feature usage counts (no content)
- Performance metrics
- Error rates (no details)
- App version and OS type

### What Telemetry NEVER Collects

- Search queries
- Document content
- File paths
- Personal identifiers
- IP addresses

### Enabling/Disabling Telemetry

```bash
# Enable (opt-in)
futurnal privacy telemetry enable

# Disable
futurnal privacy telemetry disable
```

## Cloud Features (Optional)

### Local-Only by Default

Futurnal operates entirely locally by default. No internet connection is required for core functionality.

### Optional Cloud Escalation

If you choose to enable cloud features:
- You must explicitly consent
- Clear warnings are shown before each use
- All cloud interactions are logged
- You can disable at any time

### Cloud Privacy Controls

When cloud features are enabled, you can configure:
- Maximum context sent to cloud
- Name anonymization
- Date generalization
- Provider selection

## Data Deletion

### Deleting Specific Data

```bash
# Delete data from one source
futurnal privacy data delete --source "vault:my-vault"
```

### Complete Data Deletion

```bash
# Delete all Futurnal data
futurnal privacy data delete --all
```

Or manually delete the data directory:
- **macOS/Linux**: `rm -rf ~/.futurnal`
- **Windows**: Delete `%USERPROFILE%\.futurnal`

## Security Measures

### Encryption

- Credentials: Encrypted via OS keychain
- Data at rest: Optional encryption available
- Network: HTTPS for any enabled integrations

### Access Control

- Local file permissions
- No remote access by default
- No authentication bypass

## Third-Party Components

Futurnal uses open-source components:

| Component | Purpose | Data Access |
|-----------|---------|-------------|
| Ollama | Local LLM inference | Query context only |
| ChromaDB | Vector storage | Embeddings only |
| Neo4j | Graph storage | Knowledge graph |

These components run locally and do not transmit data externally.

## Research Foundation

Futurnal's privacy design is grounded in research:

- **2501.13904v3**: "Privacy-Preserving Personalized Federated Prompt Learning" (ICLR 2025)
  - Local Differential Privacy for prompt learning
  - Balancing personalization and privacy

This research informs our approach to:
- Keeping learning local
- Preparing for future federated features
- Maintaining privacy guarantees

## Your Rights

You have the right to:

1. **Access**: View all data Futurnal has processed
2. **Export**: Export your data at any time
3. **Delete**: Permanently delete your data
4. **Consent**: Control what data is processed
5. **Audit**: Review all operations performed

## Changes to This Policy

We may update this privacy policy. Changes will be:
- Documented in the changelog
- Communicated via release notes
- Effective upon app update

## Contact

For privacy concerns or questions:
- GitHub Issues: [futurnal/futurnal/issues](https://github.com/futurnal/futurnal/issues)
- Email: privacy@futurnal.com

---

**Summary**: Futurnal processes your data locally on your device. We don't collect, transmit, or share your personal information. You have full control over your data and can delete it at any time.
