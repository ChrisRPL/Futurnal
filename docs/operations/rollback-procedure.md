# Rollback Procedure

**Futurnal v1.0.0 - Phase 1 (Archivist)**

This document describes procedures for rolling back Futurnal releases when issues are discovered.

---

## Quick Reference

| Scenario | Action | Time |
|----------|--------|------|
| Minor UI bug | Patch release | 1-2 hours |
| Data corruption | Full rollback | 30 min |
| Security issue | Emergency rollback | 15 min |
| Performance degradation | Investigation + patch | 2-4 hours |

---

## Pre-Rollback Checklist

Before initiating a rollback:

- [ ] Identify the issue severity (Critical / High / Medium / Low)
- [ ] Determine affected versions
- [ ] Assess data integrity
- [ ] Notify stakeholders if Critical
- [ ] Prepare rollback target version
- [ ] Backup current state

---

## Rollback Scenarios

### 1. Desktop Application Rollback

**When to use:** Desktop app crashes, UI issues, or feature bugs.

#### Steps:

1. **Identify the last known good version:**
   ```bash
   # Check release history
   gh release list
   ```

2. **Download previous installer:**
   - macOS: Download `.dmg` from releases
   - Windows: Download `.msi` from releases
   - Linux: Download `.AppImage` from releases

3. **Uninstall current version:**

   **macOS:**
   ```bash
   # Move to trash
   mv /Applications/Futurnal.app ~/.Trash/
   ```

   **Windows:**
   - Control Panel → Programs → Uninstall Futurnal

   **Linux:**
   ```bash
   rm ~/.local/share/applications/futurnal.AppImage
   ```

4. **Install previous version:**
   - Run the downloaded installer
   - Verify installation: `futurnal --version`

5. **Verify data integrity:**
   ```bash
   futurnal health check
   ```

---

### 2. Backend/CLI Rollback

**When to use:** Backend crashes, data processing issues, CLI bugs.

#### Steps:

1. **Check current version:**
   ```bash
   pip show futurnal
   ```

2. **Identify rollback target:**
   ```bash
   pip index versions futurnal
   ```

3. **Rollback to previous version:**
   ```bash
   pip install futurnal==X.Y.Z
   ```

4. **Verify installation:**
   ```bash
   futurnal health check
   python -c "import futurnal; print(futurnal.__version__)"
   ```

---

### 3. Database Rollback

**When to use:** Data corruption, schema issues, graph inconsistencies.

#### Neo4j Graph Database:

1. **Stop Futurnal services:**
   ```bash
   futurnal service stop
   ```

2. **Backup current state (if possible):**
   ```bash
   cp -r ~/.futurnal/neo4j ~/.futurnal/neo4j.backup.$(date +%Y%m%d)
   ```

3. **Restore from backup:**
   ```bash
   # Find latest backup
   ls -la ~/.futurnal/backups/neo4j/

   # Restore
   rm -rf ~/.futurnal/neo4j
   cp -r ~/.futurnal/backups/neo4j/YYYY-MM-DD ~/.futurnal/neo4j
   ```

4. **Verify graph integrity:**
   ```bash
   futurnal admin verify-graph
   ```

#### ChromaDB Vector Store:

1. **Backup current state:**
   ```bash
   cp -r ~/.futurnal/chroma ~/.futurnal/chroma.backup.$(date +%Y%m%d)
   ```

2. **Restore from backup:**
   ```bash
   rm -rf ~/.futurnal/chroma
   cp -r ~/.futurnal/backups/chroma/YYYY-MM-DD ~/.futurnal/chroma
   ```

3. **Verify vector store:**
   ```bash
   futurnal health chromadb
   ```

---

### 4. Configuration Rollback

**When to use:** Configuration changes caused issues.

#### Steps:

1. **Backup current config:**
   ```bash
   cp ~/.futurnal/config.yaml ~/.futurnal/config.yaml.backup
   ```

2. **Restore previous config:**
   ```bash
   # Find backups
   ls ~/.futurnal/config.yaml.backup.*

   # Restore
   cp ~/.futurnal/config.yaml.backup.TIMESTAMP ~/.futurnal/config.yaml
   ```

3. **Validate configuration:**
   ```bash
   futurnal config validate
   ```

---

## Emergency Rollback

For critical security issues or data corruption:

### Immediate Actions (15 minutes)

1. **Stop all services:**
   ```bash
   futurnal service stop --force
   pkill -f futurnal
   ```

2. **Prevent new connections:**
   ```bash
   # Block Ollama if needed
   # sudo lsof -ti:11434 | xargs kill -9
   ```

3. **Preserve state for investigation:**
   ```bash
   mkdir -p ~/.futurnal/incident-$(date +%Y%m%d)
   cp -r ~/.futurnal/audit ~/.futurnal/incident-$(date +%Y%m%d)/
   cp ~/.futurnal/config.yaml ~/.futurnal/incident-$(date +%Y%m%d)/
   ```

4. **Restore from last known good backup:**
   ```bash
   # Full workspace restore
   rm -rf ~/.futurnal/neo4j ~/.futurnal/chroma
   cp -r ~/.futurnal/backups/full/YYYY-MM-DD/* ~/.futurnal/
   ```

5. **Install previous version:**
   ```bash
   pip install futurnal==PREVIOUS_VERSION
   ```

6. **Verify system health:**
   ```bash
   futurnal health check --verbose
   ```

---

## Post-Rollback Verification

### Health Checks

```bash
# Full system health
futurnal health check

# Individual components
futurnal health ollama
futurnal health neo4j
futurnal health chromadb
futurnal health disk
futurnal health memory
```

### Data Integrity Checks

```bash
# Verify graph integrity
futurnal admin verify-graph

# Verify audit chain
futurnal privacy audit verify

# Count entities
futurnal admin stats
```

### Functional Tests

1. **Search functionality:**
   ```bash
   futurnal search "test query"
   ```

2. **Chat functionality:**
   ```bash
   futurnal chat new
   futurnal chat message "Hello"
   ```

3. **Ingestion functionality:**
   ```bash
   futurnal sources obsidian vault scan <vault_name>
   ```

---

## Backup Strategy

### Automated Backups

Configure automated backups in `~/.futurnal/config.yaml`:

```yaml
backup:
  enabled: true
  schedule: "0 2 * * *"  # Daily at 2 AM
  retention_days: 30
  include:
    - neo4j
    - chroma
    - config
    - consent
  exclude:
    - quarantine
    - cache
```

### Manual Backup

```bash
# Full backup
futurnal admin backup create

# Specific component
futurnal admin backup create --component neo4j
```

### Backup Verification

```bash
# List backups
futurnal admin backup list

# Verify backup integrity
futurnal admin backup verify YYYY-MM-DD
```

---

## Communication Template

### Internal Notification

```
Subject: [ROLLBACK] Futurnal vX.Y.Z - Issue Identified

Severity: [Critical/High/Medium/Low]
Affected Versions: vX.Y.Z
Rollback Target: vA.B.C

Issue Summary:
[Brief description of the issue]

Impact:
[Description of user impact]

Timeline:
- Issue identified: [time]
- Rollback initiated: [time]
- Rollback completed: [time]

Next Steps:
[Description of investigation/fix plans]
```

---

## Prevention

### Pre-Release Checks

1. Run full test suite
2. Run quality gate tests
3. Run security audit
4. Test on all platforms
5. Test upgrade path from previous version
6. Test rollback procedure

### Monitoring

Set up alerts for:
- Error rate spikes
- Latency increases
- Memory usage anomalies
- Disk space warnings

---

## Version History

| Version | Rollback Tested | Notes |
|---------|-----------------|-------|
| 1.0.0 | Yes | Initial release |

---

*Part of Step 10: Production Readiness*
*Phase 1 (Archivist) - December 2024*
