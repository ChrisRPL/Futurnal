//! Causal Chain IPC commands.
//!
//! Step 08: Frontend Intelligence Integration - Phase 2
//!
//! Research Foundation:
//! - Youtu-GraphRAG: Multi-hop causal reasoning
//! - CausalRAG: Causal-aware retrieval
//!
//! Handles causal chain exploration:
//! - Finding causes of an event
//! - Finding effects of an event
//! - Finding causal paths between events

use serde::{Deserialize, Serialize};
use tauri::command;

/// Causal cause result from find_causes query.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalCause {
    pub cause_id: String,
    pub cause_name: String,
    pub cause_timestamp: Option<String>,
    pub distance: i32,
    pub confidence_scores: Vec<f64>,
    pub aggregate_confidence: f64,
    pub temporal_ordering_valid: bool,
}

/// Response from find_causes command.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FindCausesResponse {
    pub success: bool,
    pub target_event_id: String,
    pub causes: Vec<CausalCause>,
    pub max_hops_requested: i32,
    pub min_confidence_requested: f64,
    pub query_time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Causal effect result from find_effects query.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalEffect {
    pub effect_id: String,
    pub effect_name: String,
    pub effect_timestamp: Option<String>,
    pub distance: i32,
    pub confidence_scores: Vec<f64>,
    pub aggregate_confidence: f64,
    pub temporal_ordering_valid: bool,
}

/// Response from find_effects command.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FindEffectsResponse {
    pub success: bool,
    pub source_event_id: String,
    pub effects: Vec<CausalEffect>,
    pub max_hops_requested: i32,
    pub min_confidence_requested: f64,
    pub query_time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Causal path data.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalPath {
    pub start_event_id: String,
    pub end_event_id: String,
    pub path: Vec<String>,
    pub causal_confidence: f64,
    pub confidence_scores: Vec<f64>,
    pub temporal_ordering_valid: bool,
    pub causal_evidence: Vec<String>,
}

/// Response from find_causal_path command.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FindCausalPathResponse {
    pub success: bool,
    pub path_found: bool,
    pub path: Option<CausalPath>,
    pub start_event_id: String,
    pub end_event_id: String,
    pub max_hops_requested: i32,
    pub query_time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Find causes of an event.
///
/// Step 08: Causal Chain Visualization
///
/// Example: "What led to this decision?"
///
/// Calls: `futurnal causal causes <event_id> --json`
#[command]
pub async fn find_causes(
    event_id: String,
    max_hops: Option<i32>,
    min_confidence: Option<f64>,
) -> Result<FindCausesResponse, String> {
    let max_hops_str = max_hops.unwrap_or(3).to_string();
    let min_conf_str = min_confidence.unwrap_or(0.6).to_string();

    let args = vec![
        "causal",
        "causes",
        &event_id,
        "--max-hops",
        &max_hops_str,
        "--min-confidence",
        &min_conf_str,
        "--json",
    ];

    let response: FindCausesResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Find causes CLI failed: {}", e);
            return Ok(FindCausesResponse {
                success: false,
                target_event_id: event_id.clone(),
                causes: vec![],
                max_hops_requested: max_hops.unwrap_or(3),
                min_confidence_requested: min_confidence.unwrap_or(0.6),
                query_time_ms: 0.0,
                error: Some(format!("Find causes failed: {}", e)),
            });
        }
    };

    log::info!(
        "Found {} causes for event '{}' in {:.1}ms",
        response.causes.len(),
        event_id,
        response.query_time_ms
    );

    Ok(response)
}

/// Find effects of an event.
///
/// Step 08: Causal Chain Visualization
///
/// Example: "What resulted from this meeting?"
///
/// Calls: `futurnal causal effects <event_id> --json`
#[command]
pub async fn find_effects(
    event_id: String,
    max_hops: Option<i32>,
    min_confidence: Option<f64>,
) -> Result<FindEffectsResponse, String> {
    let max_hops_str = max_hops.unwrap_or(3).to_string();
    let min_conf_str = min_confidence.unwrap_or(0.6).to_string();

    let args = vec![
        "causal",
        "effects",
        &event_id,
        "--max-hops",
        &max_hops_str,
        "--min-confidence",
        &min_conf_str,
        "--json",
    ];

    let response: FindEffectsResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Find effects CLI failed: {}", e);
            return Ok(FindEffectsResponse {
                success: false,
                source_event_id: event_id.clone(),
                effects: vec![],
                max_hops_requested: max_hops.unwrap_or(3),
                min_confidence_requested: min_confidence.unwrap_or(0.6),
                query_time_ms: 0.0,
                error: Some(format!("Find effects failed: {}", e)),
            });
        }
    };

    log::info!(
        "Found {} effects for event '{}' in {:.1}ms",
        response.effects.len(),
        event_id,
        response.query_time_ms
    );

    Ok(response)
}

/// Find causal path between two events.
///
/// Step 08: Causal Chain Visualization
///
/// Example: "How did event A lead to event B?"
///
/// Calls: `futurnal causal path <start_id> <end_id> --json`
#[command]
pub async fn find_causal_path(
    start_id: String,
    end_id: String,
    max_hops: Option<i32>,
) -> Result<FindCausalPathResponse, String> {
    let max_hops_str = max_hops.unwrap_or(5).to_string();

    let args = vec![
        "causal",
        "path",
        &start_id,
        &end_id,
        "--max-hops",
        &max_hops_str,
        "--json",
    ];

    let response: FindCausalPathResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Find causal path CLI failed: {}", e);
            return Ok(FindCausalPathResponse {
                success: false,
                path_found: false,
                path: None,
                start_event_id: start_id.clone(),
                end_event_id: end_id.clone(),
                max_hops_requested: max_hops.unwrap_or(5),
                query_time_ms: 0.0,
                error: Some(format!("Find causal path failed: {}", e)),
            });
        }
    };

    if response.path_found {
        log::info!(
            "Found causal path from '{}' to '{}' with {} events in {:.1}ms",
            start_id,
            end_id,
            response.path.as_ref().map(|p| p.path.len()).unwrap_or(0),
            response.query_time_ms
        );
    } else {
        log::info!(
            "No causal path found from '{}' to '{}' (searched {:.1}ms)",
            start_id,
            end_id,
            response.query_time_ms
        );
    }

    Ok(response)
}
