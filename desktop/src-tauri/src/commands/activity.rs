//! Activity Stream IPC commands.
//!
//! Step 08: Frontend Intelligence Integration - Phase 3
//!
//! Research Foundation:
//! - AgentFlow: Activity tracking patterns
//! - RLHI: User interaction history
//!
//! Provides activity stream from all sources:
//! - Audit logs
//! - Search history
//! - Chat sessions
//! - Ingestion events

use serde::{Deserialize, Serialize};
use tauri::command;

/// Activity event from the stream.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActivityEvent {
    pub id: String,
    #[serde(rename = "type")]
    pub event_type: String,
    pub category: String,
    pub title: String,
    pub description: Option<String>,
    pub timestamp: String,
    pub related_entity_ids: Vec<String>,
    pub metadata: serde_json::Value,
}

/// Response from activity list command.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActivityListResponse {
    pub success: bool,
    pub events: Vec<ActivityEvent>,
    pub total: i32,
    pub limit: i32,
    pub offset: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Get activity stream with optional filters.
///
/// Step 08: Activity Stream
///
/// Calls: `futurnal activity list --json`
#[command]
pub async fn get_activity_log(
    limit: Option<i32>,
    offset: Option<i32>,
    event_types: Option<Vec<String>>,
    date_from: Option<String>,
    date_to: Option<String>,
) -> Result<ActivityListResponse, String> {
    let limit_str = limit.unwrap_or(50).to_string();
    let offset_str = offset.unwrap_or(0).to_string();

    let mut args = vec![
        "activity".to_string(),
        "list".to_string(),
        "--limit".to_string(),
        limit_str,
        "--offset".to_string(),
        offset_str,
        "--json".to_string(),
    ];

    // Add event types filter
    if let Some(types) = event_types {
        if !types.is_empty() {
            args.push("--types".to_string());
            args.push(types.join(","));
        }
    }

    // Add date filters
    if let Some(from) = date_from {
        args.push("--from".to_string());
        args.push(from);
    }
    if let Some(to) = date_to {
        args.push("--to".to_string());
        args.push(to);
    }

    let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    let response: ActivityListResponse = match crate::python::execute_cli(&args_refs).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Activity list CLI failed: {}", e);
            return Ok(ActivityListResponse {
                success: false,
                events: vec![],
                total: 0,
                limit: limit.unwrap_or(50),
                offset: offset.unwrap_or(0),
                error: Some(format!("Activity list failed: {}", e)),
            });
        }
    };

    log::info!(
        "Retrieved {} activities (total: {})",
        response.events.len(),
        response.total
    );

    Ok(response)
}

/// Get recent activities (shortcut).
///
/// Calls: `futurnal activity recent --json`
#[command]
pub async fn get_recent_activities(
    limit: Option<i32>,
) -> Result<ActivityListResponse, String> {
    get_activity_log(limit, Some(0), None, None, None).await
}
