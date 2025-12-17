# Data Sources Guide

Connect your digital knowledge to Futurnal.

## Overview

Futurnal supports multiple data sources, each with dedicated connectors optimized for the specific format:

| Source | Status | Description |
|--------|--------|-------------|
| Obsidian | Production | Markdown vaults with wikilinks |
| IMAP Email | Production | Any IMAP-compatible email |
| GitHub | Production | Repositories, issues, PRs |
| Local Files | Beta | Generic file system scanner |

## Obsidian Vault

### Adding a Vault

**Desktop App**:
1. Settings > Data Sources > Add Source
2. Select "Obsidian Vault"
3. Browse to your vault root folder
4. Click "Add"

**CLI**:
```bash
futurnal sources obsidian vault add /path/to/vault --name "my-vault"
```

### What Gets Processed

- All `.md` files in the vault
- Frontmatter (YAML metadata)
- Wikilinks (`[[note]]`)
- Tags (`#tag`)
- Embedded images (referenced, not content)

### Sync Behavior

- **Initial**: Full scan of all files
- **Incremental**: Only changed files (via file modification time)
- **Frequency**: Configurable (default: every 5 minutes when app is open)

### Managing Vaults

```bash
# List all vaults
futurnal sources obsidian vault list

# Check vault status
futurnal sources obsidian vault status my-vault

# Scan vault manually
futurnal sources obsidian vault scan my-vault

# Remove vault (keeps files, removes from Futurnal)
futurnal sources obsidian vault remove my-vault
```

### Privacy Controls

Each vault requires explicit consent:
- Read files from the vault
- Extract metadata and content
- Store in knowledge graph

You can revoke consent at any time:
```bash
futurnal privacy consent revoke --source "vault:my-vault"
```

---

## Email (IMAP)

### Supported Providers

Any IMAP-compatible email service:
- Gmail (with App Password)
- Outlook/Hotmail
- iCloud
- FastMail
- Self-hosted IMAP servers

### Adding Email Account

**Desktop App**:
1. Settings > Data Sources > Add Source
2. Select "Email (IMAP)"
3. Enter connection details:
   - Server: `imap.gmail.com`
   - Port: `993`
   - Username: `your@email.com`
   - Password: App-specific password
4. Click "Connect"

**CLI**:
```bash
futurnal sources imap add \
  --server imap.gmail.com \
  --port 993 \
  --username your@email.com \
  --name "personal-email"
```

You'll be prompted for the password securely.

### Gmail App Password

Gmail requires an App Password:
1. Go to Google Account > Security
2. Enable 2-Factor Authentication
3. Go to App Passwords
4. Generate a password for "Futurnal"

### What Gets Processed

- Email subject and body
- Sender and recipients
- Timestamps
- Thread structure
- Attachments (text-based only)

**NOT processed**:
- Email content is never sent to cloud
- Credentials stored in system keychain
- Only metadata logged in audit

### Folder Selection

By default, all folders are synced. To select specific folders:

```bash
futurnal sources imap folders list personal-email
futurnal sources imap folders select personal-email "INBOX,Sent,Important"
```

### Managing Email Sources

```bash
# List email accounts
futurnal sources imap list

# Check sync status
futurnal sources imap status personal-email

# Sync manually
futurnal sources imap sync personal-email

# Remove account
futurnal sources imap remove personal-email
```

---

## GitHub

### Adding a Repository

**Desktop App**:
1. Settings > Data Sources > Add Source
2. Select "GitHub Repository"
3. Authenticate with GitHub (OAuth)
4. Select repositories to add

**CLI**:
```bash
# Authenticate (opens browser)
futurnal sources github auth

# Add repository
futurnal sources github add owner/repo --name "my-project"
```

### What Gets Processed

- README and documentation
- Code files (with language detection)
- Issues and issue comments
- Pull requests and PR comments
- Commit messages
- Wiki pages (if enabled)

### Sync Behavior

- **Initial**: Full repository clone
- **Incremental**: Via GitHub API (webhooks optional)
- **Rate Limits**: Respects GitHub API limits

### Managing GitHub Sources

```bash
# List repositories
futurnal sources github list

# Check sync status
futurnal sources github status my-project

# Sync manually
futurnal sources github sync my-project

# Remove repository
futurnal sources github remove my-project
```

### Private Repositories

Private repositories require appropriate GitHub OAuth scopes:
- `repo` - Full control of private repositories
- `read:org` - Read organization membership (for org repos)

---

## Local Files

### Adding a Directory

**Desktop App**:
1. Settings > Data Sources > Add Source
2. Select "Local Files"
3. Browse to directory
4. Configure file types to include

**CLI**:
```bash
futurnal sources local add /path/to/directory --name "documents"
```

### Supported File Types

| Extension | Handler |
|-----------|---------|
| `.md`, `.markdown` | Markdown parser |
| `.txt` | Plain text |
| `.pdf` | PDF extractor |
| `.docx` | Word document |
| `.html` | HTML stripper |
| `.json` | JSON content |
| `.py`, `.js`, `.ts` | Code files |

### File Filtering

Include or exclude by pattern:
```bash
futurnal sources local add /docs \
  --include "*.md,*.txt" \
  --exclude "node_modules/*,*.log"
```

### Managing Local Sources

```bash
# List local sources
futurnal sources local list

# Check status
futurnal sources local status documents

# Rescan directory
futurnal sources local scan documents

# Remove source
futurnal sources local remove documents
```

---

## Common Operations

### Viewing All Sources

**Desktop App**: Settings > Data Sources

**CLI**:
```bash
futurnal sources list
```

Output:
```
DATA SOURCES
------------
Type      Name           Status    Last Sync
--------  -------------  --------  ----------------
obsidian  my-vault       active    2024-12-17 10:30
imap      personal-email active    2024-12-17 09:15
github    my-project     syncing   2024-12-17 11:00
```

### Checking Health

```bash
futurnal sources health
```

### Viewing Quarantine

Files that fail processing are quarantined:
```bash
# View quarantined files
futurnal sources quarantine list

# Retry quarantined file
futurnal sources quarantine retry <file-id>

# Clear quarantine (delete failed files)
futurnal sources quarantine clear
```

### Consent Management

Each source requires explicit consent:
```bash
# View consent status
futurnal privacy consent list

# Revoke consent for a source
futurnal privacy consent revoke --source "vault:my-vault"

# Grant consent
futurnal privacy consent grant --source "vault:my-vault" --scope "read,process,store"
```

---

## Performance Tips

### Large Vaults (10,000+ files)

1. **Initial sync**: Run overnight for large vaults
2. **Incremental**: Enable file watching for faster updates
3. **Memory**: Allow 4GB+ RAM for processing

### Many Email Threads

1. **Limit folders**: Only sync important folders
2. **Date range**: Filter by date to reduce volume
3. **Batch size**: Adjust in settings if needed

### Large Repositories

1. **Sparse checkout**: Only sync relevant paths
2. **Shallow clone**: Limit commit history
3. **Branch filter**: Only sync specific branches

---

## Troubleshooting

### "Cannot access vault"
- Check folder permissions
- Ensure path exists
- Verify no sync conflicts (Dropbox, iCloud)

### "IMAP connection failed"
- Verify server and port
- Check credentials (use App Password for Gmail)
- Test connection: `openssl s_client -connect imap.gmail.com:993`

### "GitHub rate limit exceeded"
- Wait for rate limit reset
- Reduce sync frequency
- Use personal access token for higher limits

### "File stuck in quarantine"
- Check file for corruption
- Verify file type is supported
- Check error message: `futurnal sources quarantine info <file-id>`

---

Next: [Search & Chat Guide](search-chat.md)
