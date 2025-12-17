# Privacy Settings Guide

Complete control over your data and privacy in Futurnal.

## Privacy Philosophy

Futurnal is built on these privacy principles:

1. **Local-First**: All processing happens on your device by default
2. **Explicit Consent**: Every data access requires your permission
3. **Audit Everything**: All operations are logged (without content)
4. **No Surprises**: Clear explanations of what happens with your data
5. **Full Control**: Revoke access and delete data at any time

## Consent System

### How Consent Works

Before Futurnal can access any data source, you must grant explicit consent:

1. **Data Source Consent**: Permission to read from a specific source
2. **Processing Consent**: Permission to extract and process content
3. **Storage Consent**: Permission to store in the knowledge graph

### Viewing Consent Status

**Desktop App**: Settings > Privacy > Consent

**CLI**:
```bash
futurnal privacy consent list
```

Output:
```
CONSENT REGISTRY
----------------
Source                 Scope           Status    Granted
--------------------  --------------  --------  ----------------
vault:my-vault        read,process    granted   2024-12-01 10:00
imap:personal-email   read,process    granted   2024-12-05 14:30
github:my-project     read,process    granted   2024-12-10 09:15
```

### Granting Consent

When adding a data source, you're asked to consent. To grant manually:

```bash
futurnal privacy consent grant \
  --source "vault:my-vault" \
  --scope "read,process,store"
```

### Revoking Consent

Revoke consent at any time:

**Desktop App**: Settings > Privacy > Consent > Revoke

**CLI**:
```bash
futurnal privacy consent revoke --source "vault:my-vault"
```

**What happens when you revoke**:
- Futurnal stops accessing that source
- Existing data remains (until you delete it)
- Sync is disabled for that source

### Deleting Data After Revocation

To also delete the data:

```bash
# Revoke and delete
futurnal privacy consent revoke --source "vault:my-vault" --delete-data

# Or delete separately
futurnal privacy data delete --source "vault:my-vault"
```

---

## Audit Logging

### What Gets Logged

Futurnal maintains tamper-evident audit logs of all operations:

| Operation | Logged |
|-----------|--------|
| File accessed | Yes (path hash, not content) |
| Search executed | Yes (intent type, not query) |
| Consent changed | Yes (source, action) |
| Data deleted | Yes (scope, timestamp) |

### What is NEVER Logged

- Query content (your actual searches)
- Document content (your actual notes)
- Email content
- Credentials
- Personal identifiers

### Viewing Audit Logs

**Desktop App**: Settings > Privacy > Audit Log

**CLI**:
```bash
# View recent entries
futurnal privacy audit tail

# View by date
futurnal privacy audit show --date 2024-12-17

# Export audit log
futurnal privacy audit export --output audit.json
```

### Audit Log Format

```json
{
  "timestamp": "2024-12-17T10:30:00Z",
  "job_id": "search_1702812600000",
  "source": "search_api",
  "action": "search_executed",
  "status": "success",
  "metadata": {
    "search_type": "hybrid",
    "intent": "temporal",
    "result_count": 15,
    "latency_ms": 250
  }
}
```

### Verifying Audit Integrity

Audit logs use hash chaining for tamper detection:

```bash
futurnal privacy audit verify
```

Output:
```
Audit Log Verification
----------------------
Log file: ~/.futurnal/audit/audit.log
Entries: 1,247
Chain integrity: VALID
Last verified: 2024-12-17 11:00:00
```

---

## Data Management

### Viewing Your Data

See what Futurnal has stored:

```bash
# Summary
futurnal privacy data summary

# By source
futurnal privacy data list --source "vault:my-vault"

# Export all
futurnal privacy data export --output my-data.json
```

### Deleting Data

Delete data by source:

```bash
# Delete data from specific source
futurnal privacy data delete --source "vault:my-vault"

# Delete all data (full reset)
futurnal privacy data delete --all
```

**Warning**: Deletion is permanent. Export first if you want a backup.

### Data Retention

Configure automatic data retention:

**Desktop App**: Settings > Privacy > Retention

```yaml
# ~/.futurnal/config.yaml
privacy:
  retention:
    enabled: true
    days: 365  # Delete data older than 1 year
    exclude_sources:  # Never auto-delete these
      - "vault:permanent-notes"
```

---

## Telemetry

### What is Telemetry?

Anonymous usage statistics to improve Futurnal:
- Feature usage counts
- Performance metrics
- Error rates (no content)

### Telemetry is Opt-In

Telemetry is **disabled by default**. You must explicitly enable it.

**Desktop App**: Settings > Privacy > Telemetry > Enable

**CLI**:
```bash
futurnal privacy telemetry enable
```

### What Telemetry Collects

If enabled:
```json
{
  "event": "search_executed",
  "search_type": "temporal",
  "latency_ms": 250,
  "result_count": 15,
  "app_version": "1.0.0",
  "os": "macos",
  "timestamp": "2024-12-17"
}
```

### What Telemetry NEVER Collects

- Query content
- Document content
- File paths
- Personal identifiers
- IP addresses
- Anything that could identify you

### Disabling Telemetry

```bash
futurnal privacy telemetry disable
```

---

## Security

### Credential Storage

Credentials (email passwords, API keys) are stored securely:

| Platform | Storage |
|----------|---------|
| macOS | Keychain |
| Windows | Credential Manager |
| Linux | libsecret (GNOME Keyring) |

### Encryption

**At Rest**: Knowledge graph is stored locally without encryption by default.

To enable encryption:

```yaml
# ~/.futurnal/config.yaml
security:
  encryption:
    enabled: true
    # Key derived from your system credentials
```

### Network Security

- All local processing by default
- No external connections without consent
- HTTPS for any enabled integrations
- No data leaves your device unless explicitly enabled

### Viewing Security Status

```bash
futurnal privacy security status
```

Output:
```
Security Status
---------------
Credential storage: Keychain (macOS)
Encryption at rest: Disabled
Network connections: Local only
Cloud escalation: Disabled
Last security check: 2024-12-17 10:00
```

---

## Cloud Escalation (Optional)

### What is Cloud Escalation?

For complex queries, you can optionally use cloud LLMs. This is:
- **Disabled by default**
- Requires explicit consent
- Shows clear warnings
- Logs all cloud interactions

### Enabling Cloud Escalation

**Desktop App**: Settings > Privacy > Cloud > Enable

**CLI**:
```bash
futurnal privacy cloud enable --provider openai
```

You'll be shown:
```
WARNING: Cloud Escalation
-------------------------
This will send query context to external servers.
Provider: OpenAI
Data sent: Query + relevant context snippets
Data NOT sent: Full documents, credentials, audit logs

Do you consent? [y/N]
```

### Cloud Privacy Controls

When cloud is enabled:
```yaml
privacy:
  cloud:
    enabled: true
    provider: openai
    max_context_chars: 4000  # Limit context sent
    redact_names: true       # Anonymize names
    redact_dates: true       # Generalize dates
```

### Disabling Cloud

```bash
futurnal privacy cloud disable
```

---

## Privacy Report

Generate a complete privacy report:

```bash
futurnal privacy report --output privacy-report.html
```

The report includes:
- All consent grants
- Data sources and volumes
- Audit log summary
- Telemetry status
- Security configuration
- Recommendations

---

## Privacy Settings Summary

### Quick Commands

| Action | Command |
|--------|---------|
| View consent | `futurnal privacy consent list` |
| Revoke consent | `futurnal privacy consent revoke --source X` |
| View audit | `futurnal privacy audit tail` |
| Delete data | `futurnal privacy data delete --source X` |
| Enable telemetry | `futurnal privacy telemetry enable` |
| Security status | `futurnal privacy security status` |
| Full report | `futurnal privacy report` |

### Configuration File

All privacy settings in `~/.futurnal/config.yaml`:

```yaml
privacy:
  # Audit logging
  audit_logging: true
  audit_retention_days: 90

  # Telemetry (opt-in)
  telemetry: false

  # Local-only mode
  local_only: true

  # Cloud escalation (disabled by default)
  cloud:
    enabled: false
    provider: null

  # Data retention
  retention:
    enabled: false
    days: 365
```

---

## FAQ

**Q: Does Futurnal send my data to the cloud?**
A: No, by default. All processing is local. Cloud features are opt-in.

**Q: Can I see exactly what data Futurnal has?**
A: Yes. Use `futurnal privacy data export` for a complete export.

**Q: How do I completely remove my data?**
A: Use `futurnal privacy data delete --all` or delete `~/.futurnal`.

**Q: Is my data encrypted?**
A: By default, no. You can enable encryption in settings.

**Q: What happens if I revoke consent?**
A: Futurnal stops accessing that source. Data remains until you delete it.

---

Your privacy is fundamental to Futurnal's design. If you have questions or concerns, please open an issue on GitHub.
