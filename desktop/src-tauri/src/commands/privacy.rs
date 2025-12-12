//! Privacy and consent management IPC commands.
//!
//! Handles consent records and audit log access via Python CLI.

use serde::{Deserialize, Serialize};
use tauri::command;

/// Consent record for a data source.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConsentRecord {
    pub source_id: String,
    pub source_name: String,
    pub consent_type: String,
    pub granted: bool,
    pub granted_at: Option<String>,
    pub expires_at: Option<String>,
}

/// Request to grant consent.
#[derive(Debug, Deserialize)]
pub struct GrantConsentRequest {
    pub source_id: String,
    pub consent_type: String,
    pub duration_days: Option<u32>,
}

/// Audit log entry.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AuditLogEntry {
    pub id: String,
    pub timestamp: String,
    pub action: String,
    pub resource_type: String,
    pub resource_id: Option<String>,
    pub details: Option<serde_json::Value>,
}

/// Raw audit event from Python CLI (matches Python's AuditEvent structure).
#[derive(Debug, Deserialize)]
struct RawAuditEvent {
    timestamp: String,
    action: String,
    resource_type: Option<String>,
    resource_id: Option<String>,
    #[serde(default)]
    details: Option<serde_json::Value>,
}

/// Query parameters for audit logs.
#[derive(Debug, Deserialize)]
pub struct AuditLogQuery {
    pub limit: Option<u32>,
    pub offset: Option<u32>,
    pub action_filter: Option<String>,
    pub start_date: Option<String>,
    pub end_date: Option<String>,
}

/// Raw consent status from Python CLI.
#[derive(Debug, Deserialize)]
struct RawConsentStatus {
    source: String,
    scope: String,
    granted: bool,
    granted_at: Option<String>,
    expires_at: Option<String>,
}

/// Get consent records for a source or all sources.
///
/// Calls: `futurnal sources consent status`
#[command]
pub async fn get_consent(source_id: Option<String>) -> Result<Vec<ConsentRecord>, String> {
    if let Some(ref id) = source_id {
        log::info!("Getting consent for source: {}", id);
    } else {
        log::info!("Getting all consent records");
    }

    // Call Python CLI to get consent status
    let args = vec!["sources", "consent", "status"];

    match crate::python::execute_cli_raw(&args).await {
        Ok(output) => {
            // Parse the output - it's a text-based table, we need to convert to JSON
            // For now, return structured data based on available sources
            let mut records = Vec::new();

            // Try to parse as JSON first (in case CLI supports --json in future)
            if let Ok(statuses) = serde_json::from_str::<Vec<RawConsentStatus>>(&output) {
                for status in statuses {
                    if source_id.is_none() || source_id.as_ref() == Some(&status.source) {
                        records.push(ConsentRecord {
                            source_id: status.source.clone(),
                            source_name: status.source,
                            consent_type: status.scope,
                            granted: status.granted,
                            granted_at: status.granted_at,
                            expires_at: status.expires_at,
                        });
                    }
                }
            } else {
                // Parse text output line by line
                // Format: "Source: <name>, Scope: <scope>, Granted: <bool>, ..."
                for line in output.lines() {
                    if line.contains("Consent") || line.contains("granted") {
                        // Extract source name from context if filtering by source_id
                        if let Some(ref id) = source_id {
                            records.push(ConsentRecord {
                                source_id: id.clone(),
                                source_name: id.clone(),
                                consent_type: "local.external_processing".to_string(),
                                granted: line.to_lowercase().contains("granted"),
                                granted_at: None,
                                expires_at: None,
                            });
                        }
                    }
                }
            }

            log::info!("Retrieved {} consent records", records.len());
            Ok(records)
        }
        Err(e) => {
            log::warn!("Failed to get consent status: {}", e);
            // Return empty list on error (graceful degradation)
            Ok(vec![])
        }
    }
}

/// Grant consent for a data source.
///
/// Calls: `futurnal sources consent grant <source_name> --scope <scope> [--duration-hours <hours>]`
#[command]
pub async fn grant_consent(request: GrantConsentRequest) -> Result<ConsentRecord, String> {
    log::info!(
        "Granting {} consent for source: {}",
        request.consent_type,
        request.source_id
    );

    // Build CLI arguments
    let mut args = vec![
        "sources",
        "consent",
        "grant",
        &request.source_id,
        "--scope",
        &request.consent_type,
    ];

    // Convert days to hours for CLI
    let duration_str;
    if let Some(days) = request.duration_days {
        let hours = days * 24;
        duration_str = hours.to_string();
        args.push("--duration-hours");
        args.push(&duration_str);
    }

    match crate::python::execute_cli_void(&args).await {
        Ok(()) => {
            log::info!("Successfully granted consent for {}", request.source_id);

            // Return the consent record
            Ok(ConsentRecord {
                source_id: request.source_id.clone(),
                source_name: request.source_id,
                consent_type: request.consent_type,
                granted: true,
                granted_at: Some(chrono::Utc::now().to_rfc3339()),
                expires_at: request.duration_days.map(|days| {
                    (chrono::Utc::now() + chrono::Duration::days(days as i64)).to_rfc3339()
                }),
            })
        }
        Err(e) => {
            log::error!("Failed to grant consent: {}", e);
            Err(format!("Failed to grant consent: {}", e))
        }
    }
}

/// Revoke consent for a data source.
///
/// Calls: `futurnal sources consent revoke <source_name> --scope <scope>`
#[command]
pub async fn revoke_consent(source_id: String, consent_type: String) -> Result<(), String> {
    log::info!(
        "Revoking {} consent for source: {}",
        consent_type,
        source_id
    );

    let args = vec![
        "sources",
        "consent",
        "revoke",
        &source_id,
        "--scope",
        &consent_type,
    ];

    match crate::python::execute_cli_void(&args).await {
        Ok(()) => {
            log::info!("Successfully revoked consent for {}", source_id);
            Ok(())
        }
        Err(e) => {
            log::error!("Failed to revoke consent: {}", e);
            Err(format!("Failed to revoke consent: {}", e))
        }
    }
}

/// Get audit logs with optional filtering.
///
/// Calls: `futurnal sources audit --tail <limit>`
#[command]
pub async fn get_audit_logs(query: AuditLogQuery) -> Result<Vec<AuditLogEntry>, String> {
    log::info!(
        "Getting audit logs (limit: {:?}, offset: {:?})",
        query.limit,
        query.offset
    );

    // Build CLI arguments
    let limit = query.limit.unwrap_or(50);
    let limit_str = limit.to_string();
    let args = vec!["sources", "audit", "--tail", &limit_str];

    match crate::python::execute_cli_raw(&args).await {
        Ok(output) => {
            let mut entries = Vec::new();

            // Try to parse as JSON first
            if let Ok(raw_events) = serde_json::from_str::<Vec<RawAuditEvent>>(&output) {
                for (idx, event) in raw_events.into_iter().enumerate() {
                    entries.push(AuditLogEntry {
                        id: format!("audit-{}", idx),
                        timestamp: event.timestamp,
                        action: event.action,
                        resource_type: event.resource_type.unwrap_or_else(|| "unknown".to_string()),
                        resource_id: event.resource_id,
                        details: event.details,
                    });
                }
            } else {
                // Parse text output - each line is an audit entry
                // Format: [timestamp] action: resource_type/resource_id details
                for (idx, line) in output.lines().enumerate() {
                    if line.trim().is_empty() {
                        continue;
                    }

                    // Parse timestamp from beginning of line
                    let (timestamp, rest) = if line.starts_with('[') {
                        if let Some(end) = line.find(']') {
                            (line[1..end].to_string(), line[end+1..].trim())
                        } else {
                            (chrono::Utc::now().to_rfc3339(), line.trim())
                        }
                    } else {
                        (chrono::Utc::now().to_rfc3339(), line.trim())
                    };

                    // Extract action (first word after timestamp)
                    let parts: Vec<&str> = rest.splitn(2, ':').collect();
                    let action = parts.first().map(|s| s.trim()).unwrap_or("unknown");
                    let details_str = parts.get(1).map(|s| s.trim()).unwrap_or("");

                    entries.push(AuditLogEntry {
                        id: format!("audit-{}", idx),
                        timestamp,
                        action: action.to_string(),
                        resource_type: "audit".to_string(),
                        resource_id: None,
                        details: if details_str.is_empty() {
                            None
                        } else {
                            Some(serde_json::json!({ "message": details_str }))
                        },
                    });
                }
            }

            // Apply offset if specified
            let offset = query.offset.unwrap_or(0) as usize;
            let entries: Vec<_> = entries.into_iter().skip(offset).collect();

            log::info!("Retrieved {} audit log entries", entries.len());
            Ok(entries)
        }
        Err(e) => {
            log::warn!("Failed to get audit logs: {}", e);
            // Return empty list on error (graceful degradation)
            Ok(vec![])
        }
    }
}
