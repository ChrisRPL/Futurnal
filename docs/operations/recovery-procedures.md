# Recovery Procedures

Operational recovery procedures for Futurnal administrators and users.

## Overview

This document covers:
1. Common error scenarios and solutions
2. Service recovery procedures
3. Data recovery options
4. Emergency procedures

## Quick Reference

| Issue | Quick Fix |
|-------|-----------|
| Ollama not responding | `ollama serve` |
| Search returns nothing | Check ingestion status |
| Chat not working | Verify Ollama model |
| Ingestion stuck | Check quarantine |
| Out of disk space | Clear old data |
| Database corrupted | Restore from backup |

---

## Common Scenarios

### 1. Ollama Connection Failed

**Symptoms**:
- "Cannot connect to Ollama" error
- Chat/search not working
- Health check fails

**Diagnosis**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Check Ollama process
ps aux | grep ollama
```

**Recovery**:
```bash
# Start Ollama
ollama serve

# If port conflict, kill existing
pkill ollama
ollama serve

# Verify model is available
ollama list
ollama pull llama3.2:3b
```

**Prevention**:
- Add Ollama to system startup
- Configure as system service

---

### 2. No Search Results

**Symptoms**:
- Searches return empty
- "No results found" for known content

**Diagnosis**:
```bash
# Check source status
futurnal sources list

# Check specific vault
futurnal sources obsidian vault status my-vault

# Check for quarantined files
futurnal sources quarantine list
```

**Recovery**:
```bash
# If ingestion incomplete, wait or trigger scan
futurnal sources obsidian vault scan my-vault

# If source not added
futurnal sources obsidian vault add /path/to/vault

# If quarantine issues, retry
futurnal sources quarantine retry <file_id>
```

**Prevention**:
- Ensure full initial ingestion before searching
- Monitor ingestion progress

---

### 3. Chat Generation Failed

**Symptoms**:
- "Failed to generate response" error
- Timeout during chat
- Incomplete responses

**Diagnosis**:
```bash
# Check Ollama status
ollama ps

# Check model availability
ollama list

# Check system resources
top -l 1 | head -20
```

**Recovery**:
```bash
# Restart Ollama
pkill ollama && ollama serve

# Try smaller model
ollama pull llama3.2:3b
futurnal config set llm.model llama3.2:3b

# Clear chat session
futurnal chat --session new
```

**Prevention**:
- Use appropriately sized model for hardware
- Monitor memory usage
- Configure timeout settings

---

### 4. Ingestion Stuck

**Symptoms**:
- Ingestion progress doesn't advance
- Files stuck in processing
- High resource usage without progress

**Diagnosis**:
```bash
# Check status
futurnal sources obsidian vault status my-vault

# Check quarantine
futurnal sources quarantine list

# Check system resources
top
df -h
```

**Recovery**:
```bash
# View stuck files
futurnal sources quarantine list --status processing

# Force retry
futurnal sources quarantine retry-all

# If specific file is problematic
futurnal sources quarantine skip <file_id>

# Full rescan (last resort)
futurnal sources obsidian vault scan my-vault --full
```

**Prevention**:
- Start with smaller vaults for testing
- Monitor quarantine regularly
- Ensure sufficient disk space

---

### 5. Knowledge Graph Corruption

**Symptoms**:
- Inconsistent search results
- "Database error" messages
- Missing relationships

**Diagnosis**:
```bash
# Health check
futurnal health check --verbose

# Database integrity
futurnal privacy data summary
```

**Recovery**:
```bash
# Option 1: Repair (if available)
futurnal health repair

# Option 2: Clear and rebuild
# WARNING: This deletes all indexed data
futurnal privacy data delete --all

# Re-ingest all sources
futurnal sources obsidian vault scan my-vault --full
```

**Prevention**:
- Regular backups
- Clean shutdown (don't force kill)
- Monitor disk health

---

### 6. Out of Disk Space

**Symptoms**:
- Write errors
- Ingestion fails
- System slowdown

**Diagnosis**:
```bash
# Check disk usage
df -h

# Check Futurnal data size
du -sh ~/.futurnal
du -sh ~/.futurnal/*
```

**Recovery**:
```bash
# Clear old audit logs (keep recent)
futurnal privacy audit prune --keep-days 30

# Clear old chat sessions
futurnal chat sessions clear --older-than 30d

# Remove quarantined files
futurnal sources quarantine clear

# Remove a source if needed
futurnal sources obsidian vault remove old-vault --delete-data
```

**Prevention**:
- Configure data retention policies
- Monitor disk usage
- Use external drives for large vaults

---

### 7. Consent Issues

**Symptoms**:
- "Consent required" errors
- Source data not accessible
- Privacy errors

**Diagnosis**:
```bash
# Check consent status
futurnal privacy consent list
```

**Recovery**:
```bash
# Grant missing consent
futurnal privacy consent grant --source "vault:my-vault" --scope "read,process,store"

# If consent was revoked accidentally
futurnal privacy consent grant --source <source> --scope <scope>
```

**Prevention**:
- Document consent decisions
- Review consent periodically

---

### 8. Configuration Corruption

**Symptoms**:
- "Invalid configuration" errors
- Settings not applying
- Startup failures

**Diagnosis**:
```bash
# View current config
futurnal config show

# Check config file
cat ~/.futurnal/config.yaml
```

**Recovery**:
```bash
# Reset to defaults
futurnal config reset

# Or manually fix
# Backup current
cp ~/.futurnal/config.yaml ~/.futurnal/config.yaml.backup

# Reset
futurnal config reset

# Reapply critical settings
futurnal config set llm.model llama3.2:3b
```

**Prevention**:
- Backup config before changes
- Use CLI for config changes

---

## Service Recovery

### Full Restart Procedure

When Futurnal is not responding:

```bash
# 1. Stop all services
pkill -f futurnal
pkill ollama

# 2. Wait for clean shutdown
sleep 5

# 3. Check no orphan processes
ps aux | grep -E "futurnal|ollama"

# 4. Start Ollama
ollama serve &

# 5. Wait for Ollama ready
sleep 10

# 6. Verify Ollama
curl http://localhost:11434/api/tags

# 7. Start Futurnal health check
futurnal health check
```

### Database Recovery

If the knowledge graph is corrupted:

```bash
# 1. Backup existing data (if possible)
cp -r ~/.futurnal/data ~/.futurnal/data.backup

# 2. Check for existing backups
ls ~/.futurnal/backups/

# 3. Restore from backup (if available)
# cp -r ~/.futurnal/backups/YYYY-MM-DD/data ~/.futurnal/data

# 4. Or rebuild from scratch
futurnal privacy data delete --all
# Re-add and scan all sources
```

---

## Data Recovery

### Recovering Deleted Data

Futurnal does not have automatic recovery for deleted data. Prevention:

1. **Regular Backups**: Copy `~/.futurnal/` periodically
2. **Export Before Delete**: Use `futurnal privacy data export`
3. **Soft Delete**: Revoke consent before deleting

### Restoring from Backup

```bash
# 1. Stop Futurnal
pkill -f futurnal

# 2. Restore from backup
cp -r /path/to/backup/.futurnal ~/.futurnal

# 3. Restart and verify
futurnal health check
```

---

## Emergency Procedures

### Complete Reset

When all else fails:

```bash
# WARNING: This deletes ALL Futurnal data

# 1. Stop services
pkill -f futurnal
pkill ollama

# 2. Remove all data
rm -rf ~/.futurnal

# 3. Restart Ollama
ollama serve

# 4. Reinitialize
futurnal health check

# 5. Re-add sources
futurnal sources obsidian vault add /path/to/vault
```

### Reporting Issues

If you encounter unrecoverable issues:

1. **Collect diagnostics**:
   ```bash
   futurnal health check --verbose > diagnostics.txt
   futurnal privacy audit tail --limit 100 >> diagnostics.txt
   ```

2. **Remove sensitive data** from diagnostics

3. **Open issue** at GitHub with:
   - Steps to reproduce
   - Expected vs actual behavior
   - Diagnostics (sanitized)
   - System info (OS, memory, disk)

---

## Backup Strategy

### Recommended Backup Schedule

| Data | Frequency | Method |
|------|-----------|--------|
| Config | Weekly | Copy `~/.futurnal/config.yaml` |
| Knowledge Graph | Daily | Copy `~/.futurnal/data/` |
| Audit Logs | Monthly | Export and archive |
| Sessions | Optional | Export important sessions |

### Backup Script Example

```bash
#!/bin/bash
# backup-futurnal.sh

BACKUP_DIR="$HOME/futurnal-backups/$(date +%Y-%m-%d)"
mkdir -p "$BACKUP_DIR"

# Backup config
cp ~/.futurnal/config.yaml "$BACKUP_DIR/"

# Backup data (while Futurnal is running is OK for cold backup)
cp -r ~/.futurnal/data "$BACKUP_DIR/"

# Compress
tar -czvf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "Backup created: $BACKUP_DIR.tar.gz"
```

---

## Monitoring

### Health Check Automation

Add to crontab for regular checks:

```bash
# Check every hour
0 * * * * /usr/local/bin/futurnal health check --quiet || echo "Futurnal unhealthy" | mail -s "Alert" admin@example.com
```

### Key Metrics to Monitor

- Disk usage in `~/.futurnal/`
- Memory usage during search/chat
- Ollama response times
- Quarantine growth

---

## Contact

For issues not covered here:
- GitHub Issues: Report bugs
- Documentation: Check user guide
- Audit Logs: Review for clues
