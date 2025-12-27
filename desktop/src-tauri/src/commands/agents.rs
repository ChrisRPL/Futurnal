//! AgentFlow commands for Phase 2E.
//!
//! Provides IPC commands for:
//! - Memory buffer management
//! - Hypothesis generation and investigation
//! - Correlation verification
//! - AgentFlow status monitoring

use serde::{Deserialize, Serialize};
use tauri::command;

// ============================================================================
// Memory Buffer Types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStatsResponse {
    pub success: bool,
    #[serde(default)]
    pub total_entries: i32,
    #[serde(default)]
    pub max_entries: i32,
    #[serde(default)]
    pub utilization: f64,
    #[serde(default)]
    pub by_type: serde_json::Value,
    #[serde(default)]
    pub by_priority: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryEntry {
    pub entry_id: String,
    pub entry_type: String,
    pub content: String,
    pub priority: String,
    pub timestamp: String,
    #[serde(default)]
    pub related_entries: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
    #[serde(default)]
    pub access_count: i32,
    pub last_accessed: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryEntriesResponse {
    pub success: bool,
    #[serde(default)]
    pub entries: Vec<MemoryEntry>,
    #[serde(default)]
    pub count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemorySearchResponse {
    pub success: bool,
    pub query: Option<String>,
    #[serde(default)]
    pub entries: Vec<MemoryEntry>,
    #[serde(default)]
    pub count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryClearResponse {
    pub success: bool,
    #[serde(default)]
    pub cleared_count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ============================================================================
// Hypothesis Types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Hypothesis {
    pub hypothesis_id: String,
    pub hypothesis_type: String,
    pub description: String,
    pub event_type_a: String,
    pub event_type_b: String,
    pub confidence: f64,
    #[serde(default)]
    pub evidence_for: Vec<String>,
    #[serde(default)]
    pub evidence_against: Vec<String>,
    pub status: String,
    pub created_at: String,
    pub last_updated: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HypothesesResponse {
    pub success: bool,
    #[serde(default)]
    pub hypotheses: Vec<Hypothesis>,
    #[serde(default)]
    pub count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueryPlan {
    pub hypothesis_id: String,
    #[serde(default)]
    pub queries: Vec<serde_json::Value>,
    #[serde(default)]
    pub expected_results: Vec<String>,
    #[serde(default)]
    pub completion_criteria: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InvestigateResponse {
    pub success: bool,
    pub hypothesis_id: Option<String>,
    pub query_plan: Option<QueryPlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ============================================================================
// Verification Types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerificationReport {
    pub hypothesis_id: String,
    pub result: String,
    pub confidence: f64,
    #[serde(default)]
    pub evidence_summary: String,
    #[serde(default)]
    pub criteria_met: Vec<String>,
    #[serde(default)]
    pub criteria_violated: Vec<String>,
    #[serde(default)]
    pub recommendation: String,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerifyResponse {
    pub success: bool,
    pub hypothesis_id: Option<String>,
    pub report: Option<VerificationReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerificationHistoryResponse {
    pub success: bool,
    pub hypothesis_id: Option<String>,
    #[serde(default)]
    pub history: Vec<VerificationReport>,
    #[serde(default)]
    pub count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ============================================================================
// Status Types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentFlowStatusResponse {
    pub success: bool,
    pub memory_buffer: Option<serde_json::Value>,
    pub correlation_planner: Option<serde_json::Value>,
    pub correlation_verifier: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportPriorsResponse {
    pub success: bool,
    #[serde(default)]
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ============================================================================
// Memory Buffer Commands
// ============================================================================

/// Get memory buffer statistics.
///
/// Calls: `futurnal agents memory-stats --json`
#[command]
pub async fn get_memory_stats() -> Result<MemoryStatsResponse, String> {
    let args = vec!["agents", "memory-stats", "--json"];

    let response: MemoryStatsResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Get memory stats CLI failed: {}", e);
            return Ok(MemoryStatsResponse {
                success: false,
                total_entries: 0,
                max_entries: 50,
                utilization: 0.0,
                by_type: serde_json::json!({}),
                by_priority: serde_json::json!({}),
                error: Some(format!("Failed to get memory stats: {}", e)),
            });
        }
    };

    log::info!("Retrieved memory stats (entries={})", response.total_entries);
    Ok(response)
}

/// Get recent memory entries.
///
/// Calls: `futurnal agents memory-recent --json [--limit N]`
#[command]
pub async fn get_memory_recent(limit: Option<i32>) -> Result<MemoryEntriesResponse, String> {
    let limit_str;
    let mut args = vec!["agents", "memory-recent", "--json"];

    if let Some(l) = limit {
        limit_str = l.to_string();
        args.push("--limit");
        args.push(&limit_str);
    }

    let response: MemoryEntriesResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Get memory recent CLI failed: {}", e);
            return Ok(MemoryEntriesResponse {
                success: false,
                entries: vec![],
                count: 0,
                error: Some(format!("Failed to get recent entries: {}", e)),
            });
        }
    };

    log::info!("Retrieved {} recent memory entries", response.count);
    Ok(response)
}

/// Search memory buffer for relevant entries.
///
/// Calls: `futurnal agents memory-search <query> --json [--limit N]`
#[command]
pub async fn search_memory(query: String, limit: Option<i32>) -> Result<MemorySearchResponse, String> {
    let limit_str;
    let mut args = vec!["agents", "memory-search", &query, "--json"];

    if let Some(l) = limit {
        limit_str = l.to_string();
        args.push("--limit");
        args.push(&limit_str);
    }

    let response: MemorySearchResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Search memory CLI failed: {}", e);
            return Ok(MemorySearchResponse {
                success: false,
                query: Some(query),
                entries: vec![],
                count: 0,
                error: Some(format!("Failed to search memory: {}", e)),
            });
        }
    };

    log::info!("Memory search found {} entries", response.count);
    Ok(response)
}

/// Clear all memory entries.
///
/// Calls: `futurnal agents memory-clear --json`
#[command]
pub async fn clear_memory() -> Result<MemoryClearResponse, String> {
    let args = vec!["agents", "memory-clear", "--json"];

    let response: MemoryClearResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Clear memory CLI failed: {}", e);
            return Ok(MemoryClearResponse {
                success: false,
                cleared_count: 0,
                error: Some(format!("Failed to clear memory: {}", e)),
            });
        }
    };

    log::info!("Cleared {} memory entries", response.cleared_count);
    Ok(response)
}

// ============================================================================
// Hypothesis Commands
// ============================================================================

/// List active correlation hypotheses.
///
/// Calls: `futurnal agents hypotheses --json [--status STATUS]`
#[command]
pub async fn get_hypotheses(status: Option<String>) -> Result<HypothesesResponse, String> {
    let status_owned;
    let mut args = vec!["agents", "hypotheses", "--json"];

    if let Some(s) = status {
        status_owned = s;
        args.push("--status");
        args.push(&status_owned);
    }

    let response: HypothesesResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Get hypotheses CLI failed: {}", e);
            return Ok(HypothesesResponse {
                success: false,
                hypotheses: vec![],
                count: 0,
                error: Some(format!("Failed to get hypotheses: {}", e)),
            });
        }
    };

    log::info!("Retrieved {} hypotheses", response.count);
    Ok(response)
}

/// Generate correlation hypotheses from event types.
///
/// Calls: `futurnal agents generate-hypotheses <event_types> --json`
#[command]
pub async fn generate_hypotheses(event_types: String) -> Result<HypothesesResponse, String> {
    let args = vec!["agents", "generate-hypotheses", &event_types, "--json"];

    let response: HypothesesResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Generate hypotheses CLI failed: {}", e);
            return Ok(HypothesesResponse {
                success: false,
                hypotheses: vec![],
                count: 0,
                error: Some(format!("Failed to generate hypotheses: {}", e)),
            });
        }
    };

    log::info!("Generated {} hypotheses", response.count);
    Ok(response)
}

/// Design a query plan to investigate a hypothesis.
///
/// Calls: `futurnal agents investigate <hypothesis_id> --json`
#[command]
pub async fn investigate_hypothesis(hypothesis_id: String) -> Result<InvestigateResponse, String> {
    let args = vec!["agents", "investigate", &hypothesis_id, "--json"];

    let response: InvestigateResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Investigate hypothesis CLI failed: {}", e);
            return Ok(InvestigateResponse {
                success: false,
                hypothesis_id: Some(hypothesis_id),
                query_plan: None,
                error: Some(format!("Failed to investigate hypothesis: {}", e)),
            });
        }
    };

    log::info!("Created investigation plan for hypothesis {}", response.hypothesis_id.as_deref().unwrap_or("unknown"));
    Ok(response)
}

// ============================================================================
// Verification Commands
// ============================================================================

/// Verify a hypothesis with current evidence.
///
/// Calls: `futurnal agents verify <hypothesis_id> --json`
#[command]
pub async fn verify_hypothesis(hypothesis_id: String) -> Result<VerifyResponse, String> {
    let args = vec!["agents", "verify", &hypothesis_id, "--json"];

    let response: VerifyResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Verify hypothesis CLI failed: {}", e);
            return Ok(VerifyResponse {
                success: false,
                hypothesis_id: Some(hypothesis_id),
                report: None,
                error: Some(format!("Failed to verify hypothesis: {}", e)),
            });
        }
    };

    log::info!("Verified hypothesis {}", response.hypothesis_id.as_deref().unwrap_or("unknown"));
    Ok(response)
}

/// Get verification history for a hypothesis.
///
/// Calls: `futurnal agents verification-history <hypothesis_id> --json`
#[command]
pub async fn get_verification_history(hypothesis_id: String) -> Result<VerificationHistoryResponse, String> {
    let args = vec!["agents", "verification-history", &hypothesis_id, "--json"];

    let response: VerificationHistoryResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Get verification history CLI failed: {}", e);
            return Ok(VerificationHistoryResponse {
                success: false,
                hypothesis_id: Some(hypothesis_id),
                history: vec![],
                count: 0,
                error: Some(format!("Failed to get verification history: {}", e)),
            });
        }
    };

    log::info!("Retrieved {} verification records", response.count);
    Ok(response)
}

// ============================================================================
// Status Commands
// ============================================================================

/// Get AgentFlow system status.
///
/// Calls: `futurnal agents status --json`
#[command]
pub async fn get_agentflow_status() -> Result<AgentFlowStatusResponse, String> {
    let args = vec!["agents", "status", "--json"];

    let response: AgentFlowStatusResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Get AgentFlow status CLI failed: {}", e);
            return Ok(AgentFlowStatusResponse {
                success: false,
                memory_buffer: None,
                correlation_planner: None,
                correlation_verifier: None,
                error: Some(format!("Failed to get AgentFlow status: {}", e)),
            });
        }
    };

    log::info!("Retrieved AgentFlow status");
    Ok(response)
}

/// Export AgentFlow state as natural language for token priors.
///
/// Calls: `futurnal agents export-priors --json`
#[command]
pub async fn export_token_priors() -> Result<ExportPriorsResponse, String> {
    let args = vec!["agents", "export-priors", "--json"];

    let response: ExportPriorsResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Export token priors CLI failed: {}", e);
            return Ok(ExportPriorsResponse {
                success: false,
                content: String::new(),
                error: Some(format!("Failed to export token priors: {}", e)),
            });
        }
    };

    log::info!("Exported token priors ({} chars)", response.content.len());
    Ok(response)
}
