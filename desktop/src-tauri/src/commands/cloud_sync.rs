//! Cloud sync consent management IPC commands.
//!
//! Handles cloud sync consent operations via Python CLI.
//! This module manages consent for Firebase PKG metadata backup.

use serde::{Deserialize, Serialize};
use tauri::command;

/// Cloud sync consent scope.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum CloudSyncScope {
    #[serde(rename = "cloud:pkg:metadata_backup")]
    PkgMetadataBackup,
    #[serde(rename = "cloud:pkg:settings_backup")]
    PkgSettingsBackup,
    #[serde(rename = "cloud:search:history_sync")]
    SearchHistorySync,
}

impl CloudSyncScope {
    pub fn as_str(&self) -> &'static str {
        match self {
            CloudSyncScope::PkgMetadataBackup => "metadata_backup",
            CloudSyncScope::PkgSettingsBackup => "settings_backup",
            CloudSyncScope::SearchHistorySync => "history_sync",
        }
    }
}

/// Cloud sync consent status returned from Python backend.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CloudSyncConsentStatus {
    pub has_consent: bool,
    pub granted_scopes: Vec<String>,
    pub granted_at: Option<String>,
    pub is_syncing: bool,
    pub last_sync_at: Option<String>,
}

/// Request to grant cloud sync consent.
#[derive(Debug, Deserialize)]
pub struct GrantCloudSyncRequest {
    pub scopes: Vec<String>,
    pub operator: Option<String>,
}

/// Cloud sync audit log entry.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CloudSyncAuditEntry {
    pub id: String,
    pub timestamp: String,
    pub action: String,
    pub scope: Option<String>,
    pub nodes_affected: i32,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Query parameters for cloud sync audit logs.
#[derive(Debug, Deserialize)]
pub struct CloudSyncAuditQuery {
    pub limit: Option<u32>,
    pub action_filter: Option<String>,
}

/// Raw consent status from Python CLI JSON output.
#[derive(Debug, Deserialize)]
struct RawCloudSyncStatus {
    has_consent: bool,
    granted_scopes: Vec<String>,
    granted_at: Option<String>,
    is_syncing: Option<bool>,
    last_sync_at: Option<String>,
}

/// Get current cloud sync consent status.
///
/// Calls: `futurnal cloud-sync consent status --format json`
#[command]
pub async fn get_cloud_sync_consent() -> Result<CloudSyncConsentStatus, String> {
    log::info!("Getting cloud sync consent status");

    let args = vec!["cloud-sync", "consent", "status", "--format", "json"];

    match crate::python::execute_cli_raw(&args).await {
        Ok(output) => {
            // Try to parse JSON output
            match serde_json::from_str::<RawCloudSyncStatus>(&output) {
                Ok(raw) => {
                    log::info!(
                        "Cloud sync consent status: has_consent={}, scopes={}",
                        raw.has_consent,
                        raw.granted_scopes.len()
                    );
                    Ok(CloudSyncConsentStatus {
                        has_consent: raw.has_consent,
                        granted_scopes: raw.granted_scopes,
                        granted_at: raw.granted_at,
                        is_syncing: raw.is_syncing.unwrap_or(false),
                        last_sync_at: raw.last_sync_at,
                    })
                }
                Err(e) => {
                    log::warn!("Failed to parse cloud sync status JSON: {}", e);
                    // Return default status on parse error
                    Ok(CloudSyncConsentStatus {
                        has_consent: false,
                        granted_scopes: vec![],
                        granted_at: None,
                        is_syncing: false,
                        last_sync_at: None,
                    })
                }
            }
        }
        Err(e) => {
            log::warn!("Failed to get cloud sync consent status: {}", e);
            // Return default status on CLI error
            Ok(CloudSyncConsentStatus {
                has_consent: false,
                granted_scopes: vec![],
                granted_at: None,
                is_syncing: false,
                last_sync_at: None,
            })
        }
    }
}

/// Grant cloud sync consent for specified scopes.
///
/// Calls: `futurnal cloud-sync consent grant --scope <scope> [--scope <scope>] [--operator <op>]`
#[command]
pub async fn grant_cloud_sync_consent(request: GrantCloudSyncRequest) -> Result<CloudSyncConsentStatus, String> {
    log::info!(
        "Granting cloud sync consent for {} scopes",
        request.scopes.len()
    );

    // Build CLI arguments
    let mut args: Vec<String> = vec![
        "cloud-sync".to_string(),
        "consent".to_string(),
        "grant".to_string(),
    ];

    // Add scopes
    for scope in &request.scopes {
        args.push("--scope".to_string());
        args.push(scope.clone());
    }

    // Add operator if provided
    if let Some(ref operator) = request.operator {
        args.push("--operator".to_string());
        args.push(operator.clone());
    }

    // Convert to &str for execute_cli_void
    let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    match crate::python::execute_cli_void(&args_refs).await {
        Ok(()) => {
            log::info!("Successfully granted cloud sync consent");
            // Return updated status
            get_cloud_sync_consent().await
        }
        Err(e) => {
            log::error!("Failed to grant cloud sync consent: {}", e);
            Err(format!("Failed to grant consent: {}", e))
        }
    }
}

/// Revoke cloud sync consent.
///
/// Calls: `futurnal cloud-sync consent revoke --confirm [--operator <op>]`
///
/// IMPORTANT: This will trigger deletion of all cloud data.
#[command]
pub async fn revoke_cloud_sync_consent(operator: Option<String>) -> Result<(), String> {
    log::info!("Revoking cloud sync consent (will delete cloud data)");

    let mut args: Vec<String> = vec![
        "cloud-sync".to_string(),
        "consent".to_string(),
        "revoke".to_string(),
        "--confirm".to_string(),
    ];

    if let Some(ref op) = operator {
        args.push("--operator".to_string());
        args.push(op.clone());
    }

    let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    match crate::python::execute_cli_void(&args_refs).await {
        Ok(()) => {
            log::info!("Successfully revoked cloud sync consent");
            Ok(())
        }
        Err(e) => {
            log::error!("Failed to revoke cloud sync consent: {}", e);
            Err(format!("Failed to revoke consent: {}", e))
        }
    }
}

/// Get cloud sync audit logs.
///
/// Calls: `futurnal cloud-sync audit --tail <limit> --format json`
#[command]
pub async fn get_cloud_sync_audit_logs(query: CloudSyncAuditQuery) -> Result<Vec<CloudSyncAuditEntry>, String> {
    let limit = query.limit.unwrap_or(20);
    log::info!("Getting cloud sync audit logs (limit: {})", limit);

    let limit_str = limit.to_string();
    let mut args = vec!["cloud-sync", "audit", "--tail", &limit_str, "--format", "json"];

    if let Some(ref action) = query.action_filter {
        args.push("--action");
        args.push(action.as_str());
    }

    match crate::python::execute_cli_raw(&args).await {
        Ok(output) => {
            // Try to parse JSON array
            match serde_json::from_str::<Vec<CloudSyncAuditEntry>>(&output) {
                Ok(entries) => {
                    log::info!("Retrieved {} cloud sync audit entries", entries.len());
                    Ok(entries)
                }
                Err(e) => {
                    log::warn!("Failed to parse cloud sync audit logs: {}", e);
                    Ok(vec![])
                }
            }
        }
        Err(e) => {
            log::warn!("Failed to get cloud sync audit logs: {}", e);
            Ok(vec![])
        }
    }
}

/// Log a cloud sync event to the audit log.
///
/// Called by the frontend after sync operations to record the event.
#[command]
pub async fn log_cloud_sync_audit(
    action: String,
    success: bool,
    nodes_affected: Option<i32>,
    error_message: Option<String>,
) -> Result<(), String> {
    log::info!(
        "Logging cloud sync audit: action={}, success={}, nodes={}",
        action,
        success,
        nodes_affected.unwrap_or(0)
    );

    // For now, we'll log this via the Python CLI
    // In the future, this could write directly to a local audit store
    let success_str = if success { "success" } else { "failed" };
    let nodes_str = nodes_affected.unwrap_or(0).to_string();

    let mut args = vec![
        "cloud-sync",
        "audit",
        "log",
        "--action",
        &action,
        "--status",
        success_str,
        "--nodes",
        &nodes_str,
    ];

    let error_msg;
    if let Some(ref msg) = error_message {
        error_msg = msg.clone();
        args.push("--error");
        args.push(&error_msg);
    }

    // Note: This command may not exist yet - frontend should handle gracefully
    match crate::python::execute_cli_void(&args).await {
        Ok(()) => {
            log::info!("Successfully logged cloud sync audit event");
            Ok(())
        }
        Err(e) => {
            // Log locally but don't fail - audit logging shouldn't block sync
            log::warn!("Failed to log cloud sync audit (non-blocking): {}", e);
            Ok(())
        }
    }
}

/// Scope descriptions for UI display.
#[derive(Debug, Serialize, Clone)]
pub struct CloudSyncScopeInfo {
    pub scope: String,
    pub title: String,
    pub description: String,
    pub required: bool,
    pub default_enabled: bool,
}

/// Get scope descriptions for consent UI.
#[command]
pub async fn get_cloud_sync_scope_info() -> Result<Vec<CloudSyncScopeInfo>, String> {
    // Return static scope info (matches Python definitions)
    Ok(vec![
        CloudSyncScopeInfo {
            scope: "cloud:pkg:metadata_backup".to_string(),
            title: "Knowledge Graph Structure".to_string(),
            description: "Sync graph node labels, relationships, and timestamps (NOT document content)".to_string(),
            required: true,
            default_enabled: true,
        },
        CloudSyncScopeInfo {
            scope: "cloud:pkg:settings_backup".to_string(),
            title: "App Settings".to_string(),
            description: "Sync your Futurnal preferences and settings across devices".to_string(),
            required: false,
            default_enabled: true,
        },
        CloudSyncScopeInfo {
            scope: "cloud:search:history_sync".to_string(),
            title: "Search History".to_string(),
            description: "Sync your search queries to continue research across devices".to_string(),
            required: false,
            default_enabled: false,
        },
    ])
}
