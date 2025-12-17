# User Journey Test Checklist

**Futurnal v1.0.0 - Phase 1 (Archivist)**

Manual testing checklist for validating complete user journeys.

---

## Test Environment Setup

### Prerequisites
- [ ] Clean test machine or VM
- [ ] No previous Futurnal installation
- [ ] Ollama installed and running
- [ ] Test Obsidian vault prepared (10-50 notes)
- [ ] Screen recording enabled (for bug reports)

### Test Data
- [ ] Obsidian vault with:
  - [ ] Various note formats (lists, headers, paragraphs)
  - [ ] Wikilinks between notes
  - [ ] Frontmatter metadata
  - [ ] Date-stamped notes for temporal testing
- [ ] Test email account (for IMAP testing)

---

## Journey 1: New User Onboarding

### Step 1.1: Installation
| Platform | Action | Expected Result | Status |
|----------|--------|-----------------|--------|
| macOS ARM64 | Install .dmg | App installs to /Applications | [ ] Pass |
| macOS x64 | Install .dmg | App installs to /Applications | [ ] Pass |
| Windows x64 | Install .msi | App installs to Program Files | [ ] Pass |
| Linux x64 | Run .AppImage | App runs directly | [ ] Pass |

### Step 1.2: First Launch
- [ ] App launches without errors
- [ ] Welcome screen displayed
- [ ] Privacy notice shown before any data access
- [ ] No network connections made without consent
- [ ] Workspace created at `~/.futurnal/`

**Notes:**
```
[Record any issues or observations]
```

### Step 1.3: Privacy Acknowledgment
- [ ] Privacy policy is clearly displayed
- [ ] Local-first architecture is explained
- [ ] User must explicitly acknowledge before proceeding
- [ ] Can review policy again later in settings
- [ ] Acknowledgment is logged in audit trail

**Notes:**
```
[Record any issues or observations]
```

---

## Journey 2: Adding Data Sources

### Step 2.1: Add Obsidian Vault
- [ ] "Add Source" button is visible
- [ ] Obsidian vault option is available
- [ ] File picker opens for vault selection
- [ ] Vault is validated (contains .obsidian folder)
- [ ] Consent dialog appears with clear explanations:
  - [ ] Read: What data will be accessed
  - [ ] Process: What processing will occur
  - [ ] Store: Where data will be stored
- [ ] Can select individual consent options
- [ ] Vault appears in source list after adding

**Test Data:**
- Vault path: ____________________
- Number of notes: ____________________

**Notes:**
```
[Record any issues or observations]
```

### Step 2.2: Source Management
- [ ] Can view list of configured sources
- [ ] Can see consent status for each source
- [ ] Can modify consent for existing sources
- [ ] Can remove sources
- [ ] Removal deletes associated data

---

## Journey 3: Data Ingestion

### Step 3.1: Initial Ingestion
- [ ] Ingestion starts when source is added (or manually triggered)
- [ ] Progress indicator shows:
  - [ ] Current phase (scanning, processing, indexing)
  - [ ] Documents processed / total
  - [ ] Estimated time remaining
- [ ] Can pause/cancel ingestion
- [ ] UI remains responsive during ingestion
- [ ] Background ingestion option available

**Metrics:**
- Documents processed: ____________________
- Processing time: ____________________
- Throughput: __________ docs/sec

**Notes:**
```
[Record any issues or observations]
```

### Step 3.2: Error Handling
- [ ] Corrupt files are quarantined (not blocking)
- [ ] Unsupported files are skipped with notice
- [ ] Error count displayed
- [ ] Can view quarantined files
- [ ] Ingestion completes despite individual errors

### Step 3.3: Incremental Updates
- [ ] Add new note to vault
- [ ] Re-scan detects new note
- [ ] Only new content is processed
- [ ] Existing data is preserved

---

## Journey 4: Search Functionality

### Step 4.1: Basic Search
- [ ] Search bar is prominent and accessible
- [ ] Autocomplete/suggestions work
- [ ] Results appear within 1 second
- [ ] Results show:
  - [ ] Document title
  - [ ] Relevant snippet
  - [ ] Source information
  - [ ] Relevance score/ranking
- [ ] Can click result to see full document
- [ ] Can filter results by source

**Test Queries:**
| Query | Expected Results | Actual Results | Status |
|-------|------------------|----------------|--------|
| [Simple keyword] | Relevant notes | | [ ] |
| [Multi-word phrase] | Phrase matches | | [ ] |
| [Topic from vault] | Related notes | | [ ] |

### Step 4.2: Advanced Search
- [ ] Date range filter works
- [ ] Source filter works
- [ ] Content type filter works
- [ ] Filters can be combined
- [ ] "Clear filters" resets all

### Step 4.3: Temporal Search
- [ ] "Last week" temporal query works
- [ ] "Last month" temporal query works
- [ ] Results are chronologically relevant
- [ ] Timeline view available (if implemented)

---

## Journey 5: Chat with Knowledge

### Step 5.1: Start Chat Session
- [ ] "New Chat" button visible
- [ ] Chat interface loads
- [ ] Model selector shows available models
- [ ] Can select different LLM models

### Step 5.2: Basic Chat
- [ ] Type message and send
- [ ] Response appears with streaming
- [ ] Response includes source citations
- [ ] Sources are clickable
- [ ] Response time < 3 seconds to first token

**Test Conversations:**
| Question | Expected Behavior | Status |
|----------|-------------------|--------|
| "What do I know about [topic]?" | References vault notes | [ ] |
| "Summarize my notes from last week" | Temporal awareness | [ ] |
| "How is [topic A] related to [topic B]?" | Shows connections | [ ] |

### Step 5.3: Multi-Turn Conversation
- [ ] Follow-up questions reference context
- [ ] "Tell me more" works correctly
- [ ] Can ask about specific sources mentioned
- [ ] Context is maintained across turns
- [ ] Conversation history is preserved

### Step 5.4: Chat Settings
- [ ] Can change model mid-conversation
- [ ] Can adjust temperature (if available)
- [ ] Can clear chat history
- [ ] Can export chat

---

## Journey 6: Graph Visualization

### Step 6.1: View Knowledge Graph
- [ ] Graph view accessible
- [ ] Nodes represent entities
- [ ] Edges show relationships
- [ ] Graph renders within 500ms
- [ ] Zoom and pan controls work
- [ ] Node labels are readable

### Step 6.2: Graph Interaction
- [ ] Click node to see details
- [ ] Can filter by node type
- [ ] Can search within graph
- [ ] Connected nodes are highlighted
- [ ] Can navigate from node to source document

### Step 6.3: Graph Performance
- [ ] Handles 100+ nodes smoothly
- [ ] Handles 500+ nodes without crash
- [ ] Layout algorithm produces readable graph

---

## Journey 7: Privacy & Settings

### Step 7.1: Privacy Settings
- [ ] Privacy section in settings
- [ ] View all consent records
- [ ] Can revoke consent per source
- [ ] Can export personal data
- [ ] Can delete personal data
- [ ] Data deletion is immediate

### Step 7.2: Audit Log
- [ ] View audit log entries
- [ ] Entries show operations without content
- [ ] Can filter by date
- [ ] Can filter by action type
- [ ] "Verify chain" shows integrity status

### Step 7.3: Telemetry Settings
- [ ] Telemetry is OFF by default
- [ ] Can enable telemetry (opt-in)
- [ ] Clear explanation of what is collected
- [ ] Can view collected data
- [ ] Can export telemetry data
- [ ] Can delete telemetry data

### Step 7.4: General Settings
- [ ] Appearance settings (theme)
- [ ] LLM model configuration
- [ ] Cache settings
- [ ] Export/import configuration

---

## Journey 8: Edge Cases & Error Handling

### Step 8.1: Offline Behavior
- [ ] App launches without network
- [ ] Local search works offline
- [ ] Chat works with local Ollama
- [ ] Clear message if Ollama unavailable

### Step 8.2: Large Data Volumes
- [ ] 1000+ notes ingestion completes
- [ ] Search remains fast
- [ ] Memory usage stays reasonable (<2GB)
- [ ] No crashes or freezes

### Step 8.3: Error Recovery
- [ ] Graceful handling of:
  - [ ] Ollama connection failure
  - [ ] Database corruption
  - [ ] Disk space exhaustion
  - [ ] Memory pressure
- [ ] Error messages are user-friendly
- [ ] Recovery suggestions provided

---

## Performance Benchmarks

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| Cold start time | <10s | | [ ] |
| Search latency (p50) | <500ms | | [ ] |
| Search latency (p95) | <1000ms | | [ ] |
| Chat first token | <500ms | | [ ] |
| Chat complete | <3000ms | | [ ] |
| Graph render | <500ms | | [ ] |
| Ingestion rate | >5 docs/s | | [ ] |
| Memory usage | <2GB | | [ ] |

---

## Sign-Off

### Tester Information
- Name: ____________________
- Date: ____________________
- Platform: ____________________
- Version Tested: ____________________

### Overall Assessment
- [ ] **PASS** - All critical journeys completed successfully
- [ ] **PASS WITH ISSUES** - Minor issues found (documented below)
- [ ] **FAIL** - Critical issues prevent release

### Critical Issues Found
```
[List any critical issues that must be fixed before release]
```

### Non-Critical Issues Found
```
[List any non-critical issues for future fixes]
```

### Recommendations
```
[Any recommendations for improvement]
```

### Signature
____________________

---

*Part of Step 10: Production Readiness*
*Phase 1 (Archivist) - December 2024*
