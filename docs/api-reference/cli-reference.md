# CLI Reference

Complete command-line interface reference for Futurnal.

## Overview

```bash
futurnal [OPTIONS] COMMAND [ARGS]
```

### Global Options

| Option | Description |
|--------|-------------|
| `--help`, `-h` | Show help and exit |
| `--version`, `-v` | Show version |
| `--config PATH` | Use custom config file |
| `--verbose` | Enable verbose output |
| `--quiet` | Suppress non-error output |

## Commands

### `health`

System health and diagnostics.

```bash
futurnal health COMMAND
```

#### `health check`

Run comprehensive health check.

```bash
futurnal health check [OPTIONS]

Options:
  --json         Output as JSON
  --brief        Compact output
```

**Example**:
```bash
$ futurnal health check

Futurnal Health Check
---------------------
Ollama:       CONNECTED (localhost:11434)
Models:       llama3.2:3b available
PKG:          CONNECTED (10,432 entities)
Embeddings:   READY (all-MiniLM-L6-v2)
Storage:      OK (15.2GB free)
Privacy:      Consent registry initialized
Status:       HEALTHY
```

---

### `sources`

Data source management.

```bash
futurnal sources COMMAND
```

#### `sources list`

List all configured data sources.

```bash
futurnal sources list [OPTIONS]

Options:
  --format [table|json]  Output format (default: table)
```

#### `sources health`

Check health of all sources.

```bash
futurnal sources health
```

#### `sources quarantine`

Manage quarantined files.

```bash
futurnal sources quarantine list [--limit N]
futurnal sources quarantine retry FILE_ID
futurnal sources quarantine clear [--source NAME]
futurnal sources quarantine info FILE_ID
```

---

### `sources obsidian`

Obsidian vault management.

```bash
futurnal sources obsidian vault COMMAND
```

#### `vault add`

Add an Obsidian vault.

```bash
futurnal sources obsidian vault add PATH [OPTIONS]

Arguments:
  PATH  Path to vault root folder

Options:
  --name NAME    Vault identifier (default: folder name)
  --no-consent   Skip consent prompt (assumes granted)
```

**Example**:
```bash
futurnal sources obsidian vault add ~/Documents/my-vault --name personal
```

#### `vault list`

List registered vaults.

```bash
futurnal sources obsidian vault list
```

#### `vault status`

Check vault sync status.

```bash
futurnal sources obsidian vault status NAME
```

#### `vault scan`

Manually trigger vault scan.

```bash
futurnal sources obsidian vault scan NAME [OPTIONS]

Options:
  --full    Force full rescan (not incremental)
```

#### `vault remove`

Remove vault from Futurnal.

```bash
futurnal sources obsidian vault remove NAME [OPTIONS]

Options:
  --delete-data    Also delete indexed data
```

---

### `sources imap`

Email (IMAP) management.

```bash
futurnal sources imap COMMAND
```

#### `imap add`

Add an IMAP email account.

```bash
futurnal sources imap add [OPTIONS]

Options:
  --server HOST      IMAP server hostname
  --port PORT        IMAP port (default: 993)
  --username USER    Email username
  --name NAME        Account identifier
```

**Example**:
```bash
futurnal sources imap add \
  --server imap.gmail.com \
  --port 993 \
  --username user@gmail.com \
  --name personal
```

#### `imap list`

List email accounts.

```bash
futurnal sources imap list
```

#### `imap status`

Check sync status.

```bash
futurnal sources imap status NAME
```

#### `imap sync`

Manually sync email.

```bash
futurnal sources imap sync NAME [OPTIONS]

Options:
  --full    Full sync (not incremental)
```

#### `imap folders`

Manage folder selection.

```bash
futurnal sources imap folders list NAME
futurnal sources imap folders select NAME "FOLDER1,FOLDER2"
```

#### `imap remove`

Remove email account.

```bash
futurnal sources imap remove NAME [--delete-data]
```

---

### `sources github`

GitHub repository management.

```bash
futurnal sources github COMMAND
```

#### `github auth`

Authenticate with GitHub.

```bash
futurnal sources github auth [OPTIONS]

Options:
  --token TOKEN    Use personal access token instead of OAuth
```

#### `github add`

Add a repository.

```bash
futurnal sources github add OWNER/REPO [OPTIONS]

Options:
  --name NAME    Repository identifier
  --branch BR    Branch to sync (default: default branch)
```

**Example**:
```bash
futurnal sources github add myuser/myrepo --name my-project
```

#### `github list`

List repositories.

```bash
futurnal sources github list
```

#### `github status`

Check sync status.

```bash
futurnal sources github status NAME
```

#### `github sync`

Manually sync repository.

```bash
futurnal sources github sync NAME [--full]
```

#### `github remove`

Remove repository.

```bash
futurnal sources github remove NAME [--delete-data]
```

---

### `sources local`

Local file management.

```bash
futurnal sources local COMMAND
```

#### `local add`

Add a local directory.

```bash
futurnal sources local add PATH [OPTIONS]

Options:
  --name NAME        Directory identifier
  --include GLOB     Include patterns (comma-separated)
  --exclude GLOB     Exclude patterns (comma-separated)
```

**Example**:
```bash
futurnal sources local add ~/Documents \
  --name documents \
  --include "*.md,*.txt" \
  --exclude "*.log,node_modules/*"
```

#### `local list`

List local sources.

```bash
futurnal sources local list
```

#### `local scan`

Rescan directory.

```bash
futurnal sources local scan NAME [--full]
```

#### `local remove`

Remove local source.

```bash
futurnal sources local remove NAME [--delete-data]
```

---

### `search`

Search your knowledge.

```bash
futurnal search QUERY [OPTIONS]

Arguments:
  QUERY  Search query

Options:
  --limit N              Max results (default: 10)
  --source NAME          Filter by source
  --after DATE           Filter after date (YYYY-MM-DD)
  --before DATE          Filter before date (YYYY-MM-DD)
  --type TYPE            Filter by entity type
  --format [text|json]   Output format (default: text)
  --causal               Include causal chains
```

**Examples**:
```bash
# Basic search
futurnal search "machine learning"

# Filtered search
futurnal search "python" --source vault --limit 5

# Temporal search
futurnal search "meeting notes" --after 2024-12-01 --before 2024-12-31

# JSON output
futurnal search "project" --format json
```

---

### `chat`

Start interactive chat.

```bash
futurnal chat [OPTIONS]

Options:
  --session ID    Resume or create session with ID
  --no-stream     Disable streaming responses
```

**Example**:
```bash
$ futurnal chat --session my-session

Futurnal Chat (type /help for commands, /exit to quit)
------------------------------------------------------

You: What are my main projects?

Futurnal: Based on your knowledge graph, your main projects are:
1. Data Pipeline [Source: data-pipeline/README.md]
2. API Server [Source: api-server/overview.md]
...

You: /exit
```

#### Chat Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/exit`, `/quit` | Exit chat |
| `/clear` | Clear conversation |
| `/new` | Start new session |
| `/sources` | Show sources from last response |
| `/export` | Export conversation |
| `/session` | Show session info |

#### `chat sessions`

Manage chat sessions.

```bash
futurnal chat sessions list [--limit N]
futurnal chat sessions show ID
futurnal chat sessions delete ID
futurnal chat sessions export ID [--format json|markdown]
```

---

### `privacy`

Privacy management.

```bash
futurnal privacy COMMAND
```

#### `privacy consent`

Manage consent.

```bash
futurnal privacy consent list
futurnal privacy consent grant --source NAME --scope SCOPE
futurnal privacy consent revoke --source NAME [--delete-data]
```

**Examples**:
```bash
# View consent status
futurnal privacy consent list

# Grant consent
futurnal privacy consent grant --source "vault:my-vault" --scope "read,process,store"

# Revoke consent
futurnal privacy consent revoke --source "vault:my-vault"
```

#### `privacy audit`

View audit logs.

```bash
futurnal privacy audit tail [--limit N]
futurnal privacy audit show --date DATE
futurnal privacy audit export --output FILE
futurnal privacy audit verify
```

#### `privacy data`

Manage stored data.

```bash
futurnal privacy data summary
futurnal privacy data list --source NAME
futurnal privacy data export --output FILE
futurnal privacy data delete --source NAME
futurnal privacy data delete --all
```

#### `privacy telemetry`

Manage telemetry.

```bash
futurnal privacy telemetry status
futurnal privacy telemetry enable
futurnal privacy telemetry disable
```

#### `privacy security`

Security status.

```bash
futurnal privacy security status
```

#### `privacy report`

Generate privacy report.

```bash
futurnal privacy report --output FILE
```

---

### `config`

Configuration management.

```bash
futurnal config COMMAND
```

#### `config show`

Show current configuration.

```bash
futurnal config show [--section SECTION]
```

#### `config set`

Set configuration value.

```bash
futurnal config set KEY VALUE

Examples:
  futurnal config set llm.model llama3.1:8b
  futurnal config set privacy.telemetry false
```

#### `config reset`

Reset to defaults.

```bash
futurnal config reset [--section SECTION]
```

#### `config path`

Show config file path.

```bash
futurnal config path
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FUTURNAL_CONFIG` | Path to config file |
| `FUTURNAL_DATA_DIR` | Data directory |
| `FUTURNAL_LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `OLLAMA_HOST` | Ollama endpoint (default: localhost:11434) |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Configuration error |
| 3 | Connection error (Ollama, PKG) |
| 4 | Permission/consent error |
| 5 | Source not found |

## Shell Completion

### Bash

```bash
# Add to ~/.bashrc
eval "$(_FUTURNAL_COMPLETE=bash_source futurnal)"
```

### Zsh

```bash
# Add to ~/.zshrc
eval "$(_FUTURNAL_COMPLETE=zsh_source futurnal)"
```

### Fish

```bash
# Add to ~/.config/fish/completions/futurnal.fish
_FUTURNAL_COMPLETE=fish_source futurnal | source
```

## Examples

### Common Workflows

**Initial setup**:
```bash
futurnal health check
futurnal sources obsidian vault add ~/Documents/notes --name notes
futurnal sources obsidian vault scan notes
```

**Daily usage**:
```bash
futurnal search "yesterday's meeting"
futurnal chat --session daily
```

**Privacy review**:
```bash
futurnal privacy consent list
futurnal privacy audit tail --limit 20
futurnal privacy report --output ~/privacy-report.html
```

**Troubleshooting**:
```bash
futurnal health check --verbose
futurnal sources quarantine list
futurnal sources obsidian vault status notes
```
