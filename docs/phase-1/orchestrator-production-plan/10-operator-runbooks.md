Summary: Comprehensive operator documentation including runbooks, troubleshooting guides, and operational procedures for orchestrator management.

# 10 · Operator Runbooks & Documentation

## Purpose
Provide comprehensive operational documentation for operators managing the ingestion orchestrator, including runbooks for common tasks, troubleshooting guides, performance tuning advice, and disaster recovery procedures. Ensures operators can maintain the Ghost's experiential learning pipeline with confidence.

## Scope
- Operator guide for common tasks
- Troubleshooting playbook with solutions
- Performance tuning guide
- Disaster recovery procedures
- Monitoring and alerting setup
- Security best practices
- Operational checklists
- FAQ and common issues

## Requirements Alignment
- **Operator Console**: Documentation for monitoring, pause/resume, and manual retries
- **Observability**: Guidance on using telemetry and health checks
- **Troubleshooting**: Solutions for common operational issues
- **Best Practices**: Operational excellence guidelines

## Documentation Structure

### 1. Operator Guide
Day-to-day orchestrator management.

### 2. Troubleshooting Playbook
Common issues and solutions.

### 3. Performance Tuning
Optimization guidance.

### 4. Disaster Recovery
Recovery procedures for failures.

### 5. Monitoring & Alerting
Observability setup.

## Operator Guide

### Starting the Orchestrator
```bash
# Start orchestrator with default configuration
futurnal orchestrator start

# Start with custom configuration
futurnal orchestrator start --config /path/to/config.yaml

# Start in foreground (for debugging)
futurnal orchestrator start --foreground

# Verify orchestrator is running
futurnal orchestrator status
```

### Stopping the Orchestrator
```bash
# Graceful shutdown (waits for in-flight jobs)
futurnal orchestrator stop

# Force stop (kills running jobs)
futurnal orchestrator stop --force

# Check shutdown completed
futurnal orchestrator status
```

### Managing Sources
```bash
# List registered sources
futurnal orchestrator sources list

# Pause a source (stops scheduled jobs)
futurnal orchestrator sources pause email --reason "Maintenance"

# Resume a source
futurnal orchestrator sources resume email

# Manually trigger a source
futurnal orchestrator sources trigger notes --wait
```

### Managing Jobs
```bash
# List recent jobs
futurnal orchestrator jobs list --limit 20

# View job details
futurnal orchestrator jobs show JOB_ID

# Cancel a running job
futurnal orchestrator jobs cancel JOB_ID --reason "Duplicate"

# List failed jobs
futurnal orchestrator jobs list --status failed --since 24h
```

### Managing Quarantine
```bash
# List quarantined jobs
futurnal orchestrator quarantine list

# View quarantined job details
futurnal orchestrator quarantine show JOB_ID

# Manually retry quarantined job
futurnal orchestrator quarantine retry JOB_ID --note "Fixed permissions"

# Purge old quarantined jobs
futurnal orchestrator quarantine purge --older-than-days 90 --dry-run
futurnal orchestrator quarantine purge --older-than-days 90  # Actual purge
```

### Health Checks
```bash
# Run comprehensive health check
futurnal orchestrator health

# View detailed health information
futurnal orchestrator health --verbose

# Check specific subsystem
futurnal health check neo4j
futurnal health check chroma
```

## Troubleshooting Playbook

### Issue: Orchestrator Won't Start

**Symptoms:**
- `futurnal orchestrator start` hangs or fails
- Error message: "Failed to initialize orchestrator"

**Diagnosis:**
```bash
# Check if orchestrator is already running
futurnal orchestrator status

# Check queue database
sqlite3 ~/.futurnal/queue.db "PRAGMA integrity_check"

# Check configuration
futurnal config validate

# Check disk space
df -h ~/.futurnal

# Check logs
tail -f ~/.futurnal/logs/orchestrator.log
```

**Solutions:**
1. **Already running**: Stop existing instance first
   ```bash
   futurnal orchestrator stop --force
   futurnal orchestrator start
   ```

2. **Corrupt database**: Restore from backup
   ```bash
   cp ~/.futurnal/queue.db ~/.futurnal/queue.db.corrupt
   cp ~/.futurnal/backups/queue.db ~/.futurnal/queue.db
   futurnal orchestrator start
   ```

3. **Insufficient disk space**: Free up space
   ```bash
   futurnal orchestrator quarantine purge --older-than-days 30
   futurnal telemetry clean --older-than-days 60
   ```

### Issue: Jobs Stuck in PENDING

**Symptoms:**
- Jobs remain in PENDING state indefinitely
- Queue depth increases but jobs don't execute

**Diagnosis:**
```bash
# Check orchestrator status
futurnal orchestrator status

# Check worker utilization
futurnal orchestrator status --verbose

# Check pending jobs
futurnal orchestrator jobs list --status pending

# Check for paused sources
futurnal orchestrator sources list
```

**Solutions:**
1. **Source paused**: Resume source
   ```bash
   futurnal orchestrator sources resume SOURCE_NAME
   ```

2. **Workers exhausted**: Increase worker count
   ```yaml
   # ~/.futurnal/config/orchestrator.yaml
   workers:
     max_workers: 16  # Increase from 8
   ```

3. **Scheduled for future**: Jobs will run at scheduled time
   ```bash
   # Check scheduled_for time
   futurnal orchestrator jobs show JOB_ID
   ```

### Issue: High Failure Rate

**Symptoms:**
- Many jobs in FAILED or QUARANTINED state
- Low throughput despite available workers

**Diagnosis:**
```bash
# Check failure statistics
futurnal orchestrator telemetry failures --since 24h

# List quarantined jobs
futurnal orchestrator quarantine list

# Check quarantine reasons
futurnal orchestrator quarantine list --format json | jq '.[] | .reason' | sort | uniq -c
```

**Solutions:**
1. **Permission errors**: Fix file/folder permissions
   ```bash
   # View quarantined permission errors
   futurnal orchestrator quarantine list --reason permission_denied

   # Fix permissions on source directory
   chmod -R u+r /path/to/source
   ```

2. **Parse errors**: Update Unstructured.io or skip problematic files
   ```bash
   # View parse errors
   futurnal orchestrator quarantine list --reason parse_error

   # Manually review and purge if needed
   futurnal orchestrator quarantine purge --reason parse_error --dry-run
   ```

3. **Connector errors**: Check connector-specific logs
   ```bash
   tail -f ~/.futurnal/logs/connector-*.log
   ```

### Issue: Low Throughput

**Symptoms:**
- Throughput below ≥5 MB/s target
- Jobs complete slowly

**Diagnosis:**
```bash
# Check throughput metrics
futurnal orchestrator telemetry throughput --since 1h

# Check system resources
futurnal orchestrator status

# Check worker utilization
top -p $(pgrep -f futurnal)

# Profile a job
futurnal orchestrator jobs show JOB_ID
```

**Solutions:**
1. **CPU bottleneck**: Reduce worker count or optimize parsing
   ```yaml
   workers:
     max_workers: 4  # Reduce if CPU saturated
   ```

2. **I/O bottleneck**: Check disk performance
   ```bash
   iostat -x 1 5
   # If disk is slow, consider moving queue.db to SSD
   ```

3. **Memory pressure**: Check memory usage
   ```bash
   free -h
   # If low memory, reduce worker count
   ```

### Issue: Orchestrator Crashed

**Symptoms:**
- Orchestrator process terminated unexpectedly
- Jobs stuck in RUNNING state

**Diagnosis:**
```bash
# Check crash recovery marker
ls ~/.futurnal/.orchestrator_recovery

# Check system logs
dmesg | grep -i killed

# Check orchestrator logs
tail -100 ~/.futurnal/logs/orchestrator.log
```

**Solutions:**
1. **Restart orchestrator**: Recovery is automatic
   ```bash
   futurnal orchestrator start
   # Recovery will reset RUNNING jobs to PENDING
   ```

2. **Out of memory**: Reduce worker count
   ```yaml
   workers:
     max_workers: 4
   ```

3. **Segmentation fault**: Update dependencies
   ```bash
   pip install --upgrade unstructured
   ```

## Performance Tuning Guide

### Worker Count Tuning
```yaml
# Default: Adaptive to CPU count (max 8)
workers:
  max_workers: 8
  hardware_cap_enabled: true

# High-memory system with fast storage
workers:
  max_workers: 16  # Increase for I/O-bound workloads

# Low-memory system or CPU-bound
workers:
  max_workers: 4   # Reduce to prevent thrashing
```

### Retry Policy Tuning
```yaml
# Aggressive retry (local files)
retry:
  max_attempts: 5
  base_delay_seconds: 30
  backoff_multiplier: 1.5

# Conservative retry (network sources)
retry:
  max_attempts: 3
  base_delay_seconds: 120
  backoff_multiplier: 2.0
```

### Queue Performance
```yaml
# Frequent checkpoints (slower but safer)
queue:
  wal_mode: true
  checkpoint_interval: 50

# Less frequent checkpoints (faster but longer recovery)
queue:
  checkpoint_interval: 500
```

### Telemetry Retention
```yaml
# Minimal retention (save disk space)
telemetry:
  retention_days: 30

# Extended retention (trend analysis)
telemetry:
  retention_days: 180
```

## Disaster Recovery Procedures

### Database Corruption
```bash
# 1. Stop orchestrator
futurnal orchestrator stop --force

# 2. Verify corruption
sqlite3 ~/.futurnal/queue.db "PRAGMA integrity_check"

# 3. Backup corrupt database
mv ~/.futurnal/queue.db ~/.futurnal/queue.db.corrupt

# 4. Restore from latest backup
cp ~/.futurnal/backups/queue-2024-01-15.db ~/.futurnal/queue.db

# 5. Restart orchestrator
futurnal orchestrator start

# 6. Verify recovery
futurnal orchestrator health
```

### Lost Configuration
```bash
# 1. Stop orchestrator
futurnal orchestrator stop

# 2. Restore configuration from backup
cp ~/.futurnal/backups/orchestrator.yaml ~/.futurnal/config/orchestrator.yaml

# 3. Validate configuration
futurnal config validate

# 4. Restart orchestrator
futurnal orchestrator start
```

### Data Loss (Queue Database)
```bash
# 1. Stop orchestrator
futurnal orchestrator stop --force

# 2. Create new queue database
rm ~/.futurnal/queue.db
futurnal orchestrator start  # Creates new database

# 3. Re-trigger all sources
futurnal orchestrator sources list --format json | jq -r '.[].name' | while read source; do
  futurnal orchestrator sources trigger "$source"
done
```

## Monitoring & Alerting

### Health Check Monitoring
```bash
# Run health check every 5 minutes
*/5 * * * * futurnal orchestrator health --format json > /tmp/health.json

# Alert if health check fails
if ! futurnal orchestrator health --format json | jq -e '.status == "ok"'; then
  echo "Orchestrator health check failed" | mail -s "Alert" ops@example.com
fi
```

### Queue Depth Alerting
```bash
# Alert if queue depth exceeds threshold
QUEUE_DEPTH=$(futurnal orchestrator status --format json | jq '.queue.pending')
if [ "$QUEUE_DEPTH" -gt 1000 ]; then
  echo "Queue depth high: $QUEUE_DEPTH" | mail -s "Alert" ops@example.com
fi
```

### Throughput Monitoring
```bash
# Monitor throughput
futurnal orchestrator telemetry throughput --since 1h --format json | jq '.throughput_mbps' > /var/metrics/throughput.txt

# Alert if throughput drops below baseline
THROUGHPUT=$(cat /var/metrics/throughput.txt)
if (( $(echo "$THROUGHPUT < 5.0" | bc -l) )); then
  echo "Throughput below baseline: $THROUGHPUT MB/s" | mail -s "Alert" ops@example.com
fi
```

### Quarantine Monitoring
```bash
# Alert on high quarantine count
QUARANTINE_COUNT=$(futurnal orchestrator quarantine list --format json | jq 'length')
if [ "$QUARANTINE_COUNT" -gt 100 ]; then
  echo "High quarantine count: $QUARANTINE_COUNT" | mail -s "Alert" ops@example.com
fi
```

## Security Best Practices

### 1. Configuration Security
- Never commit configuration files with secrets to version control
- Use environment variables or keyring for sensitive values
- Restrict configuration file permissions: `chmod 600 ~/.futurnal/config/orchestrator.yaml`

### 2. Database Security
- Restrict queue database permissions: `chmod 600 ~/.futurnal/queue.db`
- Regularly backup database to secure location
- Enable WAL mode for durability

### 3. Audit Logging
- Always enable audit logging: `security.audit_logging: true`
- Regularly review audit logs for suspicious activity
- Retain audit logs for compliance requirements

### 4. Path Redaction
- Enable path redaction: `security.path_redaction: true`
- Verify paths are redacted in logs and telemetry
- Never log sensitive file contents

## Operational Checklists

### Daily Checks
- [ ] Check orchestrator status: `futurnal orchestrator status`
- [ ] Review quarantine: `futurnal orchestrator quarantine list`
- [ ] Check throughput: `futurnal orchestrator telemetry throughput --since 24h`
- [ ] Review failed jobs: `futurnal orchestrator jobs list --status failed --since 24h`

### Weekly Checks
- [ ] Run health check: `futurnal orchestrator health`
- [ ] Purge old quarantine: `futurnal orchestrator quarantine purge --older-than-days 30`
- [ ] Review telemetry trends
- [ ] Verify backup strategy

### Monthly Checks
- [ ] Update orchestrator: `pip install --upgrade futurnal`
- [ ] Review configuration: `futurnal config show`
- [ ] Test disaster recovery procedures
- [ ] Review and update documentation

## FAQ

**Q: How do I increase worker count?**
A: Edit `~/.futurnal/config/orchestrator.yaml` and set `workers.max_workers`, then restart orchestrator.

**Q: How do I retry all quarantined jobs?**
A: Use `futurnal orchestrator quarantine list --format json | jq -r '.[].job_id' | xargs -n1 futurnal orchestrator quarantine retry`

**Q: How do I backup the orchestrator database?**
A: `cp ~/.futurnal/queue.db ~/.futurnal/backups/queue-$(date +%Y-%m-%d).db`

**Q: How do I migrate to a new machine?**
A: Copy `~/.futurnal/` directory to new machine, then run `futurnal orchestrator start`

**Q: How do I debug slow job execution?**
A: Use `futurnal orchestrator jobs show JOB_ID` to view job details and duration.

## Acceptance Criteria

- ✅ Operator guide covers common tasks
- ✅ Troubleshooting playbook includes solutions
- ✅ Performance tuning guide provides optimization advice
- ✅ Disaster recovery procedures documented
- ✅ Monitoring and alerting examples provided
- ✅ Security best practices outlined
- ✅ Operational checklists included
- ✅ FAQ answers common questions
- ✅ Documentation tested by operators
- ✅ Documentation published and accessible

## Test Plan

### Documentation Tests
- `test_runbook_accuracy.py`: Verify commands work as documented
- `test_troubleshooting_procedures.py`: Test solutions resolve issues
- `test_disaster_recovery.py`: Validate recovery procedures

### User Testing
- Have operators follow runbooks and provide feedback
- Validate troubleshooting solutions with real issues
- Test disaster recovery procedures in staging

## Implementation Notes

### Documentation Format
- Markdown for readability
- Code blocks for commands
- Screenshots for UI-based operations
- Links to related documentation

### Documentation Maintenance
- Update with each release
- Incorporate operator feedback
- Version control with orchestrator code
- Review quarterly for accuracy

## Open Questions

- Should runbooks be interactive (guided workflows)?
- How to handle multi-language support for documentation?
- Should we provide video tutorials for complex procedures?
- What level of detail for troubleshooting (beginner/expert)?
- Should we integrate runbooks into CLI help system?
- How to track documentation usage and effectiveness?

## Dependencies

- Operator CLI commands (Task 04)
- Telemetry system for metrics
- Health check system
- Configuration management (Task 09)


