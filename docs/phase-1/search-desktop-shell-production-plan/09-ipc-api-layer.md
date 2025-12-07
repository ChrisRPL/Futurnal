Summary: Implement secure Tauri IPC commands bridging React frontend to Python backend services.

# 09 · IPC & API Layer

## Purpose

Create the secure Inter-Process Communication (IPC) layer using Tauri commands that bridges the React frontend with the Python backend services. This includes typed command handlers, error handling, request timeouts, and TypeScript type definitions matching Python models.

**Criticality**: CRITICAL - Foundation for all backend communication

## Scope

- Tauri invoke command handlers in Rust
- TypeScript API client with type safety
- Error handling with user-friendly messages
- Request timeout handling
- Secure IPC (no Node.js integration)
- Command implementations for:
  - Search operations (`search_query`, `get_search_history`)
  - Orchestrator status (`get_orchestrator_status`)
  - Connector management (`list_sources`, `pause_source`, `resume_source`)
  - Privacy controls (`get_consent`, `grant_consent`, `revoke_consent`)
  - Audit logging (`get_audit_logs`)
- React Query integration for caching

## Requirements Alignment

- **Architecture Principle**: "Secure IPC between frontend and backend"
- **Privacy-First**: "No unnecessary Node APIs exposed"
- **Security**: Minimal attack surface via Tauri CSP

## Component Design

### Rust Command Module Structure

```rust
// src-tauri/src/commands/mod.rs
pub mod search;
pub mod connectors;
pub mod privacy;
pub mod orchestrator;

pub use search::*;
pub use connectors::*;
pub use privacy::*;
pub use orchestrator::*;
```

### Search Commands

```rust
// src-tauri/src/commands/search.rs
use serde::{Deserialize, Serialize};
use tauri::command;
use std::process::Command;

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchQuery {
    pub query: String,
    pub top_k: Option<u32>,
    pub filters: Option<SearchFilters>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchFilters {
    pub entity_types: Option<Vec<String>>,
    pub source_types: Option<Vec<String>>,
    pub date_range: Option<DateRange>,
    pub sources: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DateRange {
    pub start: Option<String>,
    pub end: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub content: String,
    pub score: f64,
    pub confidence: f64,
    pub timestamp: Option<String>,
    pub entity_type: Option<String>,
    pub source_type: Option<String>,
    pub source_confidence: Option<f64>,
    pub causal_chain: Option<CausalChain>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CausalChain {
    pub anchor: String,
    pub causes: Vec<String>,
    pub effects: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub total: u32,
    pub query_id: String,
    pub intent: QueryIntent,
    pub execution_time_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryIntent {
    pub primary: String,
    pub temporal: Option<TemporalIntent>,
    pub causal: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TemporalIntent {
    pub range_type: String,
    pub start: Option<String>,
    pub end: Option<String>,
}

#[command]
pub async fn search_query(query: SearchQuery) -> Result<SearchResponse, String> {
    // Call Python backend via subprocess or HTTP
    let output = Command::new("python")
        .args(["-m", "futurnal.cli", "search", "--json", &query.query])
        .arg("--top-k")
        .arg(query.top_k.unwrap_or(10).to_string())
        .output()
        .map_err(|e| format!("Failed to execute search: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Search failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let response: SearchResponse = serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse search response: {}", e))?;

    Ok(response)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchHistoryItem {
    pub id: String,
    pub query: String,
    pub timestamp: String,
    pub result_count: u32,
}

#[command]
pub async fn get_search_history(limit: Option<u32>) -> Result<Vec<SearchHistoryItem>, String> {
    let output = Command::new("python")
        .args(["-m", "futurnal.cli", "search", "history", "--json"])
        .arg("--limit")
        .arg(limit.unwrap_or(50).to_string())
        .output()
        .map_err(|e| format!("Failed to get search history: {}", e))?;

    if !output.status.success() {
        return Ok(vec![]); // Return empty if no history
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let history: Vec<SearchHistoryItem> = serde_json::from_str(&stdout).unwrap_or_default();

    Ok(history)
}
```

### Connector Commands

```rust
// src-tauri/src/commands/connectors.rs
use serde::{Deserialize, Serialize};
use tauri::command;
use std::process::Command;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Connector {
    pub id: String,
    pub name: String,
    pub connector_type: String,
    pub status: String,
    pub last_sync: Option<String>,
    pub next_sync: Option<String>,
    pub progress: Option<SyncProgress>,
    pub error: Option<String>,
    pub config: serde_json::Value,
    pub stats: ConnectorStats,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SyncProgress {
    pub current: u32,
    pub total: u32,
    pub phase: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConnectorStats {
    pub files_processed: u32,
    pub entities_extracted: u32,
    pub last_duration: Option<u64>,
}

#[command]
pub async fn list_sources() -> Result<Vec<Connector>, String> {
    let output = Command::new("python")
        .args(["-m", "futurnal.cli", "sources", "list", "--json"])
        .output()
        .map_err(|e| format!("Failed to list sources: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to list sources: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let connectors: Vec<Connector> = serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse sources: {}", e))?;

    Ok(connectors)
}

#[derive(Debug, Deserialize)]
pub struct AddSourceRequest {
    pub connector_type: String,
    pub name: String,
    pub config: serde_json::Value,
}

#[command]
pub async fn add_source(request: AddSourceRequest) -> Result<Connector, String> {
    let config_str = serde_json::to_string(&request.config)
        .map_err(|e| format!("Failed to serialize config: {}", e))?;

    let output = Command::new("python")
        .args([
            "-m", "futurnal.cli", "sources", "add",
            "--type", &request.connector_type,
            "--name", &request.name,
            "--config", &config_str,
            "--json"
        ])
        .output()
        .map_err(|e| format!("Failed to add source: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to add source: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let connector: Connector = serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse connector: {}", e))?;

    Ok(connector)
}

#[command]
pub async fn pause_source(id: String) -> Result<(), String> {
    let output = Command::new("python")
        .args(["-m", "futurnal.cli", "sources", "pause", &id])
        .output()
        .map_err(|e| format!("Failed to pause source: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to pause source: {}", stderr));
    }

    Ok(())
}

#[command]
pub async fn resume_source(id: String) -> Result<(), String> {
    let output = Command::new("python")
        .args(["-m", "futurnal.cli", "sources", "resume", &id])
        .output()
        .map_err(|e| format!("Failed to resume source: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to resume source: {}", stderr));
    }

    Ok(())
}

#[command]
pub async fn delete_source(id: String) -> Result<(), String> {
    let output = Command::new("python")
        .args(["-m", "futurnal.cli", "sources", "remove", &id])
        .output()
        .map_err(|e| format!("Failed to delete source: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to delete source: {}", stderr));
    }

    Ok(())
}

#[command]
pub async fn retry_source(id: String) -> Result<(), String> {
    let output = Command::new("python")
        .args(["-m", "futurnal.cli", "sources", "retry", &id])
        .output()
        .map_err(|e| format!("Failed to retry source: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to retry source: {}", stderr));
    }

    Ok(())
}

#[command]
pub async fn pause_all_sources() -> Result<(), String> {
    let output = Command::new("python")
        .args(["-m", "futurnal.cli", "sources", "pause-all"])
        .output()
        .map_err(|e| format!("Failed to pause all sources: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to pause all sources: {}", stderr));
    }

    Ok(())
}

#[command]
pub async fn resume_all_sources() -> Result<(), String> {
    let output = Command::new("python")
        .args(["-m", "futurnal.cli", "sources", "resume-all"])
        .output()
        .map_err(|e| format!("Failed to resume all sources: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to resume all sources: {}", stderr));
    }

    Ok(())
}
```

### Privacy Commands

```rust
// src-tauri/src/commands/privacy.rs
use serde::{Deserialize, Serialize};
use tauri::command;
use std::process::Command;

#[derive(Debug, Serialize, Deserialize)]
pub struct ConsentRecord {
    pub source_id: String,
    pub source_name: String,
    pub consent_type: String,
    pub granted: bool,
    pub granted_at: Option<String>,
    pub expires_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuditLogEntry {
    pub id: String,
    pub timestamp: String,
    pub action: String,
    pub resource_type: String,
    pub resource_id: Option<String>,
    pub details: Option<serde_json::Value>,
}

#[command]
pub async fn get_consent(source_id: Option<String>) -> Result<Vec<ConsentRecord>, String> {
    let mut args = vec!["-m", "futurnal.cli", "privacy", "consent", "list", "--json"];

    let source_id_owned: String;
    if let Some(ref id) = source_id {
        source_id_owned = id.clone();
        args.push("--source");
        args.push(&source_id_owned);
    }

    let output = Command::new("python")
        .args(&args)
        .output()
        .map_err(|e| format!("Failed to get consent: {}", e))?;

    if !output.status.success() {
        return Ok(vec![]);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let consents: Vec<ConsentRecord> = serde_json::from_str(&stdout).unwrap_or_default();

    Ok(consents)
}

#[derive(Debug, Deserialize)]
pub struct GrantConsentRequest {
    pub source_id: String,
    pub consent_type: String,
    pub duration_days: Option<u32>,
}

#[command]
pub async fn grant_consent(request: GrantConsentRequest) -> Result<ConsentRecord, String> {
    let mut args = vec![
        "-m".to_string(),
        "futurnal.cli".to_string(),
        "privacy".to_string(),
        "consent".to_string(),
        "grant".to_string(),
        "--source".to_string(),
        request.source_id.clone(),
        "--type".to_string(),
        request.consent_type.clone(),
        "--json".to_string(),
    ];

    if let Some(days) = request.duration_days {
        args.push("--duration".to_string());
        args.push(days.to_string());
    }

    let output = Command::new("python")
        .args(&args)
        .output()
        .map_err(|e| format!("Failed to grant consent: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to grant consent: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let consent: ConsentRecord = serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse consent: {}", e))?;

    Ok(consent)
}

#[command]
pub async fn revoke_consent(source_id: String, consent_type: String) -> Result<(), String> {
    let output = Command::new("python")
        .args([
            "-m", "futurnal.cli", "privacy", "consent", "revoke",
            "--source", &source_id,
            "--type", &consent_type,
        ])
        .output()
        .map_err(|e| format!("Failed to revoke consent: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to revoke consent: {}", stderr));
    }

    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct AuditLogQuery {
    pub limit: Option<u32>,
    pub offset: Option<u32>,
    pub action_filter: Option<String>,
    pub start_date: Option<String>,
    pub end_date: Option<String>,
}

#[command]
pub async fn get_audit_logs(query: AuditLogQuery) -> Result<Vec<AuditLogEntry>, String> {
    let mut args = vec![
        "-m".to_string(),
        "futurnal.cli".to_string(),
        "privacy".to_string(),
        "audit".to_string(),
        "list".to_string(),
        "--json".to_string(),
    ];

    if let Some(limit) = query.limit {
        args.push("--limit".to_string());
        args.push(limit.to_string());
    }

    if let Some(offset) = query.offset {
        args.push("--offset".to_string());
        args.push(offset.to_string());
    }

    if let Some(action) = &query.action_filter {
        args.push("--action".to_string());
        args.push(action.clone());
    }

    let output = Command::new("python")
        .args(&args)
        .output()
        .map_err(|e| format!("Failed to get audit logs: {}", e))?;

    if !output.status.success() {
        return Ok(vec![]);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let logs: Vec<AuditLogEntry> = serde_json::from_str(&stdout).unwrap_or_default();

    Ok(logs)
}
```

### Orchestrator Commands

```rust
// src-tauri/src/commands/orchestrator.rs
use serde::{Deserialize, Serialize};
use tauri::command;
use std::process::Command;

#[derive(Debug, Serialize, Deserialize)]
pub struct OrchestratorStatus {
    pub running: bool,
    pub active_jobs: u32,
    pub pending_jobs: u32,
    pub failed_jobs: u32,
    pub sources: Vec<SourceStatus>,
    pub last_activity: Option<String>,
    pub uptime_seconds: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SourceStatus {
    pub id: String,
    pub name: String,
    pub status: String,
    pub progress: Option<SyncProgress>,
    pub last_sync: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SyncProgress {
    pub current: u32,
    pub total: u32,
    pub phase: String,
}

#[command]
pub async fn get_orchestrator_status() -> Result<OrchestratorStatus, String> {
    let output = Command::new("python")
        .args(["-m", "futurnal.cli", "orchestrator", "status", "--json"])
        .output()
        .map_err(|e| format!("Failed to get orchestrator status: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to get status: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let status: OrchestratorStatus = serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse status: {}", e))?;

    Ok(status)
}
```

### Knowledge Graph Commands

```rust
// src-tauri/src/commands/graph.rs
use serde::{Deserialize, Serialize};
use tauri::command;
use std::process::Command;

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub timestamp: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphLink {
    pub source: String,
    pub target: String,
    pub relationship: String,
    pub weight: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    pub links: Vec<GraphLink>,
}

#[command]
pub async fn get_knowledge_graph(limit: Option<u32>) -> Result<GraphData, String> {
    let output = Command::new("python")
        .args([
            "-m", "futurnal.cli", "graph", "export",
            "--limit", &limit.unwrap_or(1000).to_string(),
            "--json"
        ])
        .output()
        .map_err(|e| format!("Failed to get graph: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to get graph: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let graph: GraphData = serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse graph: {}", e))?;

    Ok(graph)
}
```

### Main Tauri Entry Point

```rust
// src-tauri/src/main.rs
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;

use tauri::Manager;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            // Search
            commands::search::search_query,
            commands::search::get_search_history,
            // Connectors
            commands::connectors::list_sources,
            commands::connectors::add_source,
            commands::connectors::pause_source,
            commands::connectors::resume_source,
            commands::connectors::delete_source,
            commands::connectors::retry_source,
            commands::connectors::pause_all_sources,
            commands::connectors::resume_all_sources,
            // Privacy
            commands::privacy::get_consent,
            commands::privacy::grant_consent,
            commands::privacy::revoke_consent,
            commands::privacy::get_audit_logs,
            // Orchestrator
            commands::orchestrator::get_orchestrator_status,
            // Graph
            commands::graph::get_knowledge_graph,
        ])
        .setup(|app| {
            #[cfg(debug_assertions)]
            {
                let window = app.get_webview_window("main").unwrap();
                window.open_devtools();
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### TypeScript API Client

```typescript
// src/lib/api.ts
import { invoke } from '@tauri-apps/api/core';
import type {
  SearchQuery,
  SearchResponse,
  SearchHistoryItem,
  Connector,
  AddSourceRequest,
  ConsentRecord,
  GrantConsentRequest,
  AuditLogEntry,
  AuditLogQuery,
  OrchestratorStatus,
  GraphData,
} from '@/types/api';

// Timeout wrapper
async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number = 30000
): Promise<T> {
  const timeout = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error('Request timeout')), timeoutMs);
  });
  return Promise.race([promise, timeout]);
}

// Error handler
function handleError(error: unknown): never {
  if (error instanceof Error) {
    throw new ApiError(error.message);
  }
  throw new ApiError(String(error));
}

export class ApiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

// Search API
export const searchApi = {
  async search(query: SearchQuery): Promise<SearchResponse> {
    try {
      return await withTimeout(invoke('search_query', { query }));
    } catch (error) {
      handleError(error);
    }
  },

  async getHistory(limit?: number): Promise<SearchHistoryItem[]> {
    try {
      return await withTimeout(invoke('get_search_history', { limit }));
    } catch (error) {
      return []; // Graceful fallback
    }
  },
};

// Connectors API
export const connectorsApi = {
  async list(): Promise<Connector[]> {
    try {
      return await withTimeout(invoke('list_sources'));
    } catch (error) {
      handleError(error);
    }
  },

  async add(request: AddSourceRequest): Promise<Connector> {
    try {
      return await withTimeout(invoke('add_source', { request }));
    } catch (error) {
      handleError(error);
    }
  },

  async pause(id: string): Promise<void> {
    try {
      await invoke('pause_source', { id });
    } catch (error) {
      handleError(error);
    }
  },

  async resume(id: string): Promise<void> {
    try {
      await invoke('resume_source', { id });
    } catch (error) {
      handleError(error);
    }
  },

  async delete(id: string): Promise<void> {
    try {
      await invoke('delete_source', { id });
    } catch (error) {
      handleError(error);
    }
  },

  async retry(id: string): Promise<void> {
    try {
      await invoke('retry_source', { id });
    } catch (error) {
      handleError(error);
    }
  },

  async pauseAll(): Promise<void> {
    try {
      await invoke('pause_all_sources');
    } catch (error) {
      handleError(error);
    }
  },

  async resumeAll(): Promise<void> {
    try {
      await invoke('resume_all_sources');
    } catch (error) {
      handleError(error);
    }
  },
};

// Privacy API
export const privacyApi = {
  async getConsent(sourceId?: string): Promise<ConsentRecord[]> {
    try {
      return await withTimeout(invoke('get_consent', { sourceId }));
    } catch (error) {
      return [];
    }
  },

  async grantConsent(request: GrantConsentRequest): Promise<ConsentRecord> {
    try {
      return await withTimeout(invoke('grant_consent', { request }));
    } catch (error) {
      handleError(error);
    }
  },

  async revokeConsent(sourceId: string, consentType: string): Promise<void> {
    try {
      await invoke('revoke_consent', { sourceId, consentType });
    } catch (error) {
      handleError(error);
    }
  },

  async getAuditLogs(query: AuditLogQuery): Promise<AuditLogEntry[]> {
    try {
      return await withTimeout(invoke('get_audit_logs', { query }));
    } catch (error) {
      return [];
    }
  },
};

// Orchestrator API
export const orchestratorApi = {
  async getStatus(): Promise<OrchestratorStatus> {
    try {
      return await withTimeout(invoke('get_orchestrator_status'));
    } catch (error) {
      handleError(error);
    }
  },
};

// Graph API
export const graphApi = {
  async getGraph(limit?: number): Promise<GraphData> {
    try {
      return await withTimeout(invoke('get_knowledge_graph', { limit }));
    } catch (error) {
      handleError(error);
    }
  },
};
```

### TypeScript Types

```typescript
// src/types/api.ts

// Search types
export interface SearchQuery {
  query: string;
  top_k?: number;
  filters?: SearchFilters;
}

export interface SearchFilters {
  entity_types?: string[];
  source_types?: string[];
  date_range?: DateRange;
  sources?: string[];
}

export interface DateRange {
  start?: string;
  end?: string;
}

export interface SearchResult {
  id: string;
  content: string;
  score: number;
  confidence: number;
  timestamp?: string;
  entity_type?: 'Event' | 'Document' | 'Code' | 'Person';
  source_type?: 'text' | 'ocr' | 'audio' | 'code';
  source_confidence?: number;
  causal_chain?: CausalChain;
  metadata: Record<string, unknown>;
}

export interface CausalChain {
  anchor: string;
  causes: string[];
  effects: string[];
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  query_id: string;
  intent: QueryIntent;
  execution_time_ms: number;
}

export interface QueryIntent {
  primary: 'temporal' | 'causal' | 'exploratory';
  temporal?: TemporalIntent;
  causal?: boolean;
}

export interface TemporalIntent {
  range_type: string;
  start?: string;
  end?: string;
}

export interface SearchHistoryItem {
  id: string;
  query: string;
  timestamp: string;
  result_count: number;
}

// Connector types
export interface Connector {
  id: string;
  name: string;
  type: 'local_folder' | 'obsidian' | 'github' | 'imap';
  status: 'active' | 'paused' | 'error' | 'syncing' | 'disabled';
  lastSync?: string;
  nextSync?: string;
  progress?: SyncProgress;
  error?: string;
  config: Record<string, unknown>;
  stats: ConnectorStats;
}

export interface SyncProgress {
  current: number;
  total: number;
  phase: string;
}

export interface ConnectorStats {
  filesProcessed: number;
  entitiesExtracted: number;
  lastDuration?: number;
}

export interface AddSourceRequest {
  type: Connector['type'];
  name: string;
  config: Record<string, unknown>;
}

// Privacy types
export interface ConsentRecord {
  source_id: string;
  source_name: string;
  consent_type: string;
  granted: boolean;
  granted_at?: string;
  expires_at?: string;
}

export interface GrantConsentRequest {
  source_id: string;
  consent_type: string;
  duration_days?: number;
}

export interface AuditLogEntry {
  id: string;
  timestamp: string;
  action: string;
  resource_type: string;
  resource_id?: string;
  details?: Record<string, unknown>;
}

export interface AuditLogQuery {
  limit?: number;
  offset?: number;
  action_filter?: string;
  start_date?: string;
  end_date?: string;
}

// Orchestrator types
export interface OrchestratorStatus {
  running: boolean;
  active_jobs: number;
  pending_jobs: number;
  failed_jobs: number;
  sources: SourceStatus[];
  last_activity?: string;
  uptime_seconds: number;
}

export interface SourceStatus {
  id: string;
  name: string;
  status: string;
  progress?: SyncProgress;
  last_sync?: string;
  error?: string;
}

// Graph types
export interface GraphNode {
  id: string;
  label: string;
  type: 'Event' | 'Document' | 'Person' | 'Code' | 'Concept';
  timestamp?: string;
  metadata?: Record<string, unknown>;
}

export interface GraphLink {
  source: string;
  target: string;
  relationship: string;
  weight?: number;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}
```

### React Query Hooks

```typescript
// src/hooks/useApi.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { searchApi, connectorsApi, privacyApi, orchestratorApi, graphApi } from '@/lib/api';
import type { SearchQuery, AddSourceRequest, GrantConsentRequest, AuditLogQuery } from '@/types/api';

// Search hooks
export function useSearch(query: SearchQuery, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: ['search', query],
    queryFn: () => searchApi.search(query),
    enabled: options?.enabled !== false && !!query.query,
  });
}

export function useSearchHistory(limit?: number) {
  return useQuery({
    queryKey: ['searchHistory', limit],
    queryFn: () => searchApi.getHistory(limit),
  });
}

// Connector hooks
export function useConnectors() {
  return useQuery({
    queryKey: ['connectors'],
    queryFn: connectorsApi.list,
    refetchInterval: 5000, // Poll every 5s for status updates
  });
}

export function useAddConnector() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: AddSourceRequest) => connectorsApi.add(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['connectors'] });
    },
  });
}

export function usePauseConnector() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => connectorsApi.pause(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['connectors'] });
    },
  });
}

export function useResumeConnector() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => connectorsApi.resume(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['connectors'] });
    },
  });
}

// Privacy hooks
export function useConsent(sourceId?: string) {
  return useQuery({
    queryKey: ['consent', sourceId],
    queryFn: () => privacyApi.getConsent(sourceId),
  });
}

export function useGrantConsent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: GrantConsentRequest) => privacyApi.grantConsent(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['consent'] });
    },
  });
}

export function useAuditLogs(query: AuditLogQuery) {
  return useQuery({
    queryKey: ['auditLogs', query],
    queryFn: () => privacyApi.getAuditLogs(query),
  });
}

// Orchestrator hooks
export function useOrchestratorStatus() {
  return useQuery({
    queryKey: ['orchestratorStatus'],
    queryFn: orchestratorApi.getStatus,
    refetchInterval: 3000, // Poll every 3s
  });
}

// Graph hooks
export function useKnowledgeGraph(limit?: number) {
  return useQuery({
    queryKey: ['knowledgeGraph', limit],
    queryFn: () => graphApi.getGraph(limit),
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
}
```

## Acceptance Criteria

- [ ] All Tauri commands compile and execute
- [ ] TypeScript types match Rust structures
- [ ] Error messages are user-friendly
- [ ] Request timeouts handled gracefully
- [ ] React Query hooks cache appropriately
- [ ] No Node.js APIs exposed (security)
- [ ] Search queries return results
- [ ] Connector operations work (pause, resume, delete)
- [ ] Privacy consent flows work
- [ ] Audit logs retrievable

## Test Plan

### Integration Tests
```typescript
describe('IPC Commands', () => {
  it('should execute search query', async () => {
    const result = await invoke('search_query', {
      query: { query: 'test', top_k: 10 }
    });
    expect(result.results).toBeDefined();
  });

  it('should handle timeout gracefully', async () => {
    // Test with very short timeout
    await expect(
      withTimeout(slowOperation(), 100)
    ).rejects.toThrow('Request timeout');
  });
});
```

### Mock Tests
```typescript
describe('API Client', () => {
  beforeEach(() => {
    vi.mock('@tauri-apps/api/core', () => ({
      invoke: vi.fn(),
    }));
  });

  it('should handle search errors', async () => {
    vi.mocked(invoke).mockRejectedValueOnce('Backend error');

    await expect(searchApi.search({ query: 'test' }))
      .rejects.toThrow(ApiError);
  });
});
```

## Dependencies

- @tauri-apps/api (core invoke)
- @tanstack/react-query
- serde / serde_json (Rust)
- tauri-plugin-store
- tauri-plugin-dialog

## Security Considerations

- No `nodeIntegration` in Tauri window
- Commands validate input before processing
- Sensitive data (tokens) stored securely
- Audit logging for privacy operations
- Rate limiting on API calls

## Next Steps

After IPC layer complete:
1. Add WebSocket for real-time status updates
2. Implement request retry logic
3. Add offline mode handling
4. Create API documentation

**This IPC layer provides secure, typed communication between the desktop frontend and Python backend.**
