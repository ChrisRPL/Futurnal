//! Notification IPC commands.
//!
//! Phase 2D: Notification System - Frontend integration for:
//! - Viewing and updating notification preferences
//! - Getting notification history
//! - Managing DND schedules

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tauri::command;

// ============================================================================
// Preferences Types
// ============================================================================

/// Do-not-disturb schedule.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct DndSchedule {
    pub enabled: bool,
    pub start_time: String,
    pub end_time: String,
    pub days: Vec<i32>,
    pub is_active: bool,
}

/// Channel preferences.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ChannelPreferences {
    pub dashboard_enabled: bool,
    pub desktop_enabled: bool,
    pub min_priority_desktop: String,
}

/// Notification preferences response.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NotificationPreferencesResponse {
    pub success: bool,
    pub frequency: String,
    pub max_daily_notifications: i32,
    pub min_insight_confidence: f64,
    pub dnd_schedule: DndSchedule,
    pub channels: ChannelPreferences,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Get notification preferences.
///
/// Calls: `futurnal notifications preferences --json`
#[command]
pub async fn get_notification_preferences() -> Result<NotificationPreferencesResponse, String> {
    let args = vec!["notifications", "preferences", "--json"];

    let response: NotificationPreferencesResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Get notification preferences CLI failed: {}", e);
            return Ok(NotificationPreferencesResponse {
                success: false,
                frequency: "daily".to_string(),
                max_daily_notifications: 10,
                min_insight_confidence: 0.6,
                dnd_schedule: DndSchedule {
                    enabled: true,
                    start_time: "22:00".to_string(),
                    end_time: "08:00".to_string(),
                    days: vec![0, 1, 2, 3, 4, 5, 6],
                    is_active: false,
                },
                channels: ChannelPreferences {
                    dashboard_enabled: true,
                    desktop_enabled: true,
                    min_priority_desktop: "high".to_string(),
                },
                error: Some(format!("Failed to get preferences: {}", e)),
            });
        }
    };

    log::info!("Retrieved notification preferences (frequency={})", response.frequency);

    Ok(response)
}

/// Request to update notification frequency.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetFrequencyRequest {
    pub frequency: String,
}

/// Response from setting frequency.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetFrequencyResponse {
    pub success: bool,
    pub frequency: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Set notification frequency.
///
/// Calls: `futurnal notifications set-frequency <frequency> --json`
#[command]
pub async fn set_notification_frequency(
    request: SetFrequencyRequest,
) -> Result<SetFrequencyResponse, String> {
    let args = vec!["notifications", "set-frequency", &request.frequency, "--json"];

    let response: SetFrequencyResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Set notification frequency CLI failed: {}", e);
            return Ok(SetFrequencyResponse {
                success: false,
                frequency: request.frequency,
                error: Some(format!("Failed to set frequency: {}", e)),
            });
        }
    };

    log::info!("Set notification frequency to: {}", response.frequency);

    Ok(response)
}

/// Request to update DND schedule.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetDndRequest {
    pub enabled: Option<bool>,
    pub start_time: Option<String>,
    pub end_time: Option<String>,
}

/// Response from setting DND.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetDndResponse {
    pub success: bool,
    pub dnd_schedule: Option<DndSchedule>,
    pub is_active: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Set do-not-disturb schedule.
///
/// Calls: `futurnal notifications set-dnd --json [options]`
#[command]
pub async fn set_notification_dnd(
    request: SetDndRequest,
) -> Result<SetDndResponse, String> {
    let mut args = vec!["notifications".to_string(), "set-dnd".to_string(), "--json".to_string()];

    if let Some(enabled) = request.enabled {
        if enabled {
            args.push("--enabled".to_string());
        } else {
            args.push("--disabled".to_string());
        }
    }

    if let Some(start) = &request.start_time {
        args.push("--start".to_string());
        args.push(start.clone());
    }

    if let Some(end) = &request.end_time {
        args.push("--end".to_string());
        args.push(end.clone());
    }

    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    let response: SetDndResponse = match crate::python::execute_cli(&args_ref).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Set notification DND CLI failed: {}", e);
            return Ok(SetDndResponse {
                success: false,
                dnd_schedule: None,
                is_active: false,
                error: Some(format!("Failed to set DND: {}", e)),
            });
        }
    };

    log::info!("Updated DND schedule (active={})", response.is_active);

    Ok(response)
}

// ============================================================================
// History Types
// ============================================================================

/// A notification in the history.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NotificationItem {
    pub notification_id: String,
    pub title: String,
    pub body: String,
    pub insight_id: Option<String>,
    pub priority: String,
    pub created_at: String,
    pub delivered_at: Option<String>,
    pub read: bool,
    pub action_url: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Response from getting notification history.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NotificationHistoryResponse {
    pub success: bool,
    pub notifications: Vec<NotificationItem>,
    pub total_count: i32,
    pub unread_count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Get notification history.
///
/// Calls: `futurnal notifications history --json [--limit N] [--unread]`
#[command]
pub async fn get_notification_history(
    limit: Option<i32>,
    unread_only: Option<bool>,
) -> Result<NotificationHistoryResponse, String> {
    let mut args = vec!["notifications".to_string(), "history".to_string(), "--json".to_string()];

    if let Some(l) = limit {
        args.push("--limit".to_string());
        args.push(l.to_string());
    }

    if unread_only.unwrap_or(false) {
        args.push("--unread".to_string());
    }

    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    let response: NotificationHistoryResponse = match crate::python::execute_cli(&args_ref).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Get notification history CLI failed: {}", e);
            return Ok(NotificationHistoryResponse {
                success: false,
                notifications: vec![],
                total_count: 0,
                unread_count: 0,
                error: Some(format!("Failed to get history: {}", e)),
            });
        }
    };

    log::info!(
        "Retrieved {} notifications ({} unread)",
        response.total_count,
        response.unread_count
    );

    Ok(response)
}

/// Mark a notification as read.
///
/// Calls: `futurnal notifications mark-read <notification_id> --json`
#[command]
pub async fn mark_notification_read(
    notification_id: String,
) -> Result<bool, String> {
    let args = vec!["notifications", "mark-read", &notification_id, "--json"];

    match crate::python::execute_cli::<serde_json::Value>(&args).await {
        Ok(_) => {
            log::info!("Marked notification {} as read", notification_id);
            Ok(true)
        }
        Err(e) => {
            log::warn!("Mark notification read failed: {}", e);
            Ok(false)
        }
    }
}

/// Clear all notifications.
///
/// Calls: `futurnal notifications clear --json`
#[command]
pub async fn clear_notifications() -> Result<i32, String> {
    let args = vec!["notifications", "clear", "--json"];

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct ClearResponse {
        cleared_count: i32,
    }

    match crate::python::execute_cli::<ClearResponse>(&args).await {
        Ok(resp) => {
            log::info!("Cleared {} notifications", resp.cleared_count);
            Ok(resp.cleared_count)
        }
        Err(e) => {
            log::warn!("Clear notifications failed: {}", e);
            Ok(0)
        }
    }
}

// ============================================================================
// Status Types
// ============================================================================

/// Notification system status.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NotificationStatusResponse {
    pub success: bool,
    pub pending_insights: i32,
    pub unread_notifications: i32,
    pub notifications_today: i32,
    pub max_daily: i32,
    pub frequency: String,
    pub dnd_active: bool,
    pub desktop_enabled: bool,
    pub dashboard_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Get notification system status.
///
/// Calls: `futurnal notifications status --json`
#[command]
pub async fn get_notification_status() -> Result<NotificationStatusResponse, String> {
    let args = vec!["notifications", "status", "--json"];

    let response: NotificationStatusResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Get notification status CLI failed: {}", e);
            return Ok(NotificationStatusResponse {
                success: false,
                pending_insights: 0,
                unread_notifications: 0,
                notifications_today: 0,
                max_daily: 10,
                frequency: "daily".to_string(),
                dnd_active: false,
                desktop_enabled: true,
                dashboard_enabled: true,
                error: Some(format!("Failed to get status: {}", e)),
            });
        }
    };

    log::info!(
        "Notification status: {} pending, {} unread, DND={}",
        response.pending_insights,
        response.unread_notifications,
        response.dnd_active
    );

    Ok(response)
}

/// Deliver pending notifications.
///
/// Calls: `futurnal notifications deliver --json [--force]`
#[command]
pub async fn deliver_notifications(force: Option<bool>) -> Result<i32, String> {
    let mut args = vec!["notifications".to_string(), "deliver".to_string(), "--json".to_string()];

    if force.unwrap_or(false) {
        args.push("--force".to_string());
    }

    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct DeliverResponse {
        delivered_count: i32,
    }

    match crate::python::execute_cli::<DeliverResponse>(&args_ref).await {
        Ok(resp) => {
            log::info!("Delivered {} notifications", resp.delivered_count);
            Ok(resp.delivered_count)
        }
        Err(e) => {
            log::warn!("Deliver notifications failed: {}", e);
            Ok(0)
        }
    }
}
