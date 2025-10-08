# GitHub Connector Manual Testing Checklist

This checklist covers manual verification procedures that cannot be fully automated, particularly for OAuth flows and real GitHub integration scenarios.

## Pre-Testing Setup

### Prerequisites
- [ ] Futurnal installed and configured
- [ ] Python 3.11+ environment active
- [ ] GitHub account for testing (use test account, not production)
- [ ] Network connectivity to github.com

### Test Credentials
- [ ] Create GitHub OAuth App for testing
  - Navigate to GitHub Settings → Developer settings → OAuth Apps
  - Create new OAuth App with callback URL
  - Note Client ID and Client Secret
- [ ] Generate Personal Access Token for testing
  - Navigate to GitHub Settings → Developer settings → Personal access tokens
  - Create token with `repo` scope
  - **Security**: Store token securely, revoke after testing

## 1. OAuth Device Flow Testing

### 1.1 Initiate OAuth Flow
```bash
futurnal sources github oauth start
```

**Verification:**
- [ ] User code displayed (format: XXXX-XXXX)
- [ ] Verification URL displayed (https://github.com/login/device)
- [ ] Expiration time shown (typically 15 minutes)
- [ ] No credentials logged to console

### 1.2 User Authorization
- [ ] Open verification URL in browser
- [ ] Enter displayed user code
- [ ] Authorize application access
- [ ] Verify requested scopes are appropriate

### 1.3 Token Receipt
**Expected behavior:**
- [ ] CLI polls for authorization
- [ ] Success message displayed upon authorization
- [ ] Token stored securely (not displayed)
- [ ] Token retrievable for API calls
- [ ] No token appears in logs

**Verify token storage:**
```bash
# Token should NOT be visible in any output
futurnal sources github list-credentials
```

### 1.4 Token Refresh
- [ ] Wait for token expiration (or simulate)
- [ ] Trigger sync operation
- [ ] Verify automatic token refresh
- [ ] Verify refresh token used correctly
- [ ] No disruption to ongoing operations

## 2. Repository Sync Validation

### 2.1 Small Repository Sync

**Test Repository:** Use `octocat/Hello-World` (public)

```bash
futurnal sources github add octocat/Hello-World --visibility public
futurnal sources github sync octocat/Hello-World
```

**Verification:**
- [ ] Repository registered successfully
- [ ] Sync completes in <10 seconds
- [ ] All files processed
- [ ] No errors in logs
- [ ] Elements delivered to PKG
- [ ] Sync state saved correctly

**Check results:**
```bash
futurnal sources github status octocat/Hello-World
```

- [ ] File count matches GitHub
- [ ] Latest commit SHA matches
- [ ] Sync timestamp recent

### 2.2 Medium Repository Sync

**Test Repository:** Choose repo with 100-1000 files

```bash
futurnal sources github add <owner>/<repo> --visibility private
futurnal sources github sync <owner>/<repo>
```

**Verification:**
- [ ] Sync completes in <60 seconds
- [ ] No rate limit errors
- [ ] Progress updates shown
- [ ] Memory usage acceptable (<500MB)
- [ ] API requests optimized (<100 requests)

**Monitor performance:**
```bash
# In separate terminal
watch -n 1 'ps aux | grep futurnal'
```

### 2.3 Incremental Sync

**Setup:**
1. Perform initial full sync
2. Make small change to repository (edit README)
3. Push change to GitHub

**Test:**
```bash
futurnal sources github sync <owner>/<repo>
```

**Verification:**
- [ ] Sync detects only changed files
- [ ] Sync completes in <5 seconds
- [ ] Delta processing correct
- [ ] Unchanged files not re-processed
- [ ] Sync state updated

## 3. Webhook Configuration Testing

### 3.1 Webhook Setup
```bash
futurnal sources github webhook configure <owner>/<repo>
```

**Verification:**
- [ ] Webhook created on GitHub
- [ ] Secret generated and stored
- [ ] Webhook URL correct
- [ ] Events configured (push, pull_request, issues)
- [ ] Webhook active

**Manual check on GitHub:**
- [ ] Navigate to repository settings → Webhooks
- [ ] Verify webhook exists
- [ ] Check recent deliveries

### 3.2 Webhook Event Processing

**Trigger push event:**
1. Make commit to repository
2. Push to GitHub
3. Observe webhook processing

**Verification:**
- [ ] Webhook received within seconds
- [ ] Signature validated correctly
- [ ] Sync triggered automatically
- [ ] Only changed files processed
- [ ] Webhook delivery marked successful on GitHub

**Check webhook logs:**
```bash
futurnal sources github webhook logs
```

### 3.3 Webhook Failure Handling

**Test invalid signature:**
- [ ] Manually send webhook with wrong signature
- [ ] Verify rejection (403 Forbidden)
- [ ] No processing occurs
- [ ] Security event logged

## 4. Issue/PR Metadata Extraction

### 4.1 Issue Extraction

**Setup:** Create test issue on GitHub

**Test:**
```bash
futurnal sources github sync-issues <owner>/<repo>
```

**Verification:**
- [ ] Issue fetched successfully
- [ ] Metadata normalized correctly
- [ ] Title, body, labels extracted
- [ ] Author information captured
- [ ] Timestamps accurate
- [ ] Triples generated for relationships

**Validate in PKG:**
```bash
# Query PKG for issue node
futurnal query "MATCH (i:Issue) WHERE i.number = <issue_number> RETURN i"
```

### 4.2 Pull Request Extraction

**Setup:** Create test PR on GitHub

**Test:**
```bash
futurnal sources github sync-prs <owner>/<repo>
```

**Verification:**
- [ ] PR fetched successfully
- [ ] Files changed extracted
- [ ] Review comments captured
- [ ] Merge status correct
- [ ] Branch references accurate
- [ ] Relationships to issues captured

## 5. Secret Detection Validation

### 5.1 Positive Detection

**Create test file with secrets:**
```python
# test_secrets.py
GITHUB_TOKEN = "ghp_1234567890123456789012345678901234"
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
API_SECRET = "sk_live_1234567890abcdefghijklmnop"
```

**Test:**
```bash
# Attempt to sync repository with secrets file
futurnal sources github sync <owner>/<repo>
```

**Verification:**
- [ ] Secrets detected before processing
- [ ] Files with secrets quarantined
- [ ] Warning displayed to user
- [ ] Secrets NOT ingested into PKG
- [ ] No secrets in logs
- [ ] Quarantine reason clear

### 5.2 False Positive Check

**Create test file that should pass:**
```python
# test_clean.py
def generate_token():
    """Generate secure random token."""
    return secrets.token_hex(32)

api_url = "https://api.example.com"
config = {"debug": True}
```

**Verification:**
- [ ] File processed normally
- [ ] No false positive detection
- [ ] Content ingested correctly

## 6. Consent and Privacy Validation

### 6.1 Consent Workflow

**Test consent requirement:**
```bash
# Attempt sync without consent
futurnal sources github add <owner>/<repo> --no-consent
futurnal sources github sync <owner>/<repo>
```

**Verification:**
- [ ] Sync blocked due to missing consent
- [ ] Clear error message
- [ ] Instructions for granting consent
- [ ] No data processed

**Grant consent:**
```bash
futurnal consent grant github:repo:access <repository_id>
futurnal sources github sync <owner>/<repo>
```

**Verification:**
- [ ] Sync proceeds after consent
- [ ] Consent logged in audit trail

### 6.2 Path Anonymization

**Enable strict privacy:**
```bash
futurnal sources github add <owner>/<repo> --privacy-level strict
futurnal sources github sync <owner>/<repo>
```

**Verification:**
- [ ] Full file paths NOT in logs
- [ ] Paths anonymized/hashed
- [ ] File content NOT in audit logs
- [ ] Only metadata logged

**Check audit logs:**
```bash
cat ~/.futurnal/audit/*.jsonl | grep -i "<repository_name>"
```

- [ ] No sensitive paths visible
- [ ] No content visible

## 7. Force Push Handling

### 7.1 Simulate Force Push

**Setup:**
1. Sync repository initial state
2. Note commit SHA
3. Force push with `git push --force` (rewrite history)

**Test:**
```bash
futurnal sources github sync <owner>/<repo>
```

**Verification:**
- [ ] Force push detected
- [ ] Full resync triggered (not incremental)
- [ ] Warning logged
- [ ] New history processed completely
- [ ] Old state replaced

## 8. Failure Recovery

### 8.1 Quarantine Recovery

**Setup:** File that fails processing (corrupt or unsupported format)

**Test:**
```bash
futurnal sources github sync <owner>/<repo>
futurnal sources github quarantine list
futurnal sources github quarantine retry <file_id>
```

**Verification:**
- [ ] Failed file quarantined
- [ ] Sync completes for other files
- [ ] Quarantine contains file
- [ ] Retry mechanism works
- [ ] Successful retry removes from quarantine

### 8.2 API Failure Recovery

**Simulate:**
1. Disconnect network during sync
2. Observe behavior

**Verification:**
- [ ] Graceful failure handling
- [ ] Partial progress saved
- [ ] Resume capability after reconnection
- [ ] Circuit breaker triggered after failures
- [ ] Exponential backoff applied

## 9. Provider-Specific Testing

### 9.1 GitHub.com Testing
- [ ] OAuth flow works
- [ ] 5000 req/hour rate limit respected
- [ ] GraphQL API v4 endpoints work
- [ ] REST API fallback functional

### 9.2 GitHub Enterprise Testing

**If GitHub Enterprise available:**

```bash
futurnal sources github add <owner>/<repo> \
  --host github.enterprise.com \
  --api-url https://github.enterprise.com/api/v3
```

**Verification:**
- [ ] Custom host configured
- [ ] Custom OAuth endpoints work
- [ ] API compatibility verified
- [ ] Rate limits match enterprise config

## 10. Security Compliance

### 10.1 Credential Security Audit

**Check for credential leakage:**

```bash
# Search all logs for tokens
grep -r "ghp_" ~/.futurnal/logs/ && echo "❌ LEAK DETECTED" || echo "✅ No leaks"
grep -r "gho_" ~/.futurnal/logs/ && echo "❌ LEAK DETECTED" || echo "✅ No leaks"

# Search audit logs
grep -r "ghp_" ~/.futurnal/audit/ && echo "❌ LEAK DETECTED" || echo "✅ No leaks"
```

**Verification:**
- [ ] No tokens in application logs
- [ ] No tokens in audit logs
- [ ] No tokens in exception messages
- [ ] No tokens in console output

### 10.2 HTTPS Verification

**Test:**
```bash
# Monitor network traffic (requires tcpdump/wireshark)
futurnal sources github sync <owner>/<repo>
```

**Verification:**
- [ ] All GitHub API calls use HTTPS
- [ ] No HTTP fallback attempted
- [ ] Certificate validation enabled
- [ ] TLS 1.2+ used

## Post-Testing Cleanup

- [ ] Revoke test Personal Access Tokens
- [ ] Delete test OAuth applications
- [ ] Remove test repositories
- [ ] Clear test data from PKG
- [ ] Delete quarantined test files
- [ ] Review and archive test logs

## Issue Reporting

If any tests fail:

1. **Document the failure:**
   - Step number from this checklist
   - Expected vs actual behavior
   - Error messages/logs
   - Environment details

2. **Collect diagnostics:**
   ```bash
   futurnal health check
   futurnal sources github diagnostics
   ```

3. **Create GitHub issue:**
   - Use template: Bug Report
   - Attach diagnostic output
   - Reference this checklist

## Sign-Off

Testing completed by: ________________
Date: ________________
All tests passed: ☐ Yes ☐ No
Issues filed: ________________
