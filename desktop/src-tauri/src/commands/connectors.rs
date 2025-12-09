//! Connector management IPC commands.
//!
//! Handles data source (connector) management operations.

use serde::{Deserialize, Serialize};
use tauri::command;

/// Data source connector information.
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

/// Sync progress for an active sync.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SyncProgress {
    pub current: u32,
    pub total: u32,
    pub phase: String,
}

/// Statistics for a connector.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConnectorStats {
    pub files_processed: u32,
    pub entities_extracted: u32,
    pub last_duration: Option<u64>,
}

/// Request to add a new data source.
#[derive(Debug, Deserialize)]
pub struct AddSourceRequest {
    pub connector_type: String,
    pub name: String,
    pub config: serde_json::Value,
}

/// List all configured data sources.
///
/// Calls: `futurnal sources list --json`
#[command]
pub async fn list_sources() -> Result<Vec<Connector>, String> {
    // Return empty list until CLI is wired
    // This is a real empty response - no sources configured yet
    Ok(vec![])

    // TODO: Wire to Python CLI
    // let args = vec!["sources", "list", "--json"];
    // let connectors: Vec<Connector> = crate::python::execute_cli(&args).await?;
    // Ok(connectors)
}

/// Add a new data source.
///
/// Calls: `futurnal sources add --type <type> --name <name> --config <json> --json`
#[command]
pub async fn add_source(request: AddSourceRequest) -> Result<Connector, String> {
    log::info!(
        "Adding source: {} ({})",
        request.name,
        request.connector_type
    );

    // Return error until CLI is wired - cannot create source without backend
    Err("Source creation requires Python backend. Please ensure futurnal CLI is installed.".to_string())

    // TODO: Wire to Python CLI
    // let config_str = serde_json::to_string(&request.config)
    //     .map_err(|e| format!("Failed to serialize config: {}", e))?;
    // let args = vec![
    //     "sources",
    //     "add",
    //     "--type",
    //     &request.connector_type,
    //     "--name",
    //     &request.name,
    //     "--config",
    //     &config_str,
    //     "--json",
    // ];
    // let connector: Connector = crate::python::execute_cli(&args).await?;
    // Ok(connector)
}

/// Pause a data source.
///
/// Calls: `futurnal sources pause <id>`
#[command]
pub async fn pause_source(id: String) -> Result<(), String> {
    log::info!("Pausing source: {}", id);

    // Return success for now - operation is valid but no-op without sources
    Ok(())

    // TODO: Wire to Python CLI
    // let args = vec!["sources", "pause", &id];
    // crate::python::execute_cli_void(&args).await?;
    // Ok(())
}

/// Resume a paused data source.
///
/// Calls: `futurnal sources resume <id>`
#[command]
pub async fn resume_source(id: String) -> Result<(), String> {
    log::info!("Resuming source: {}", id);

    Ok(())

    // TODO: Wire to Python CLI
    // let args = vec!["sources", "resume", &id];
    // crate::python::execute_cli_void(&args).await?;
    // Ok(())
}

/// Delete a data source.
///
/// Calls: `futurnal sources remove <id>`
#[command]
pub async fn delete_source(id: String) -> Result<(), String> {
    log::info!("Deleting source: {}", id);

    Ok(())

    // TODO: Wire to Python CLI
    // let args = vec!["sources", "remove", &id];
    // crate::python::execute_cli_void(&args).await?;
    // Ok(())
}

/// Retry a failed source sync.
///
/// Calls: `futurnal sources quarantine retry <id>`
#[command]
pub async fn retry_source(id: String) -> Result<(), String> {
    log::info!("Retrying source: {}", id);

    Ok(())

    // TODO: Wire to Python CLI
    // let args = vec!["sources", "quarantine", "retry", &id];
    // crate::python::execute_cli_void(&args).await?;
    // Ok(())
}

/// Pause all data sources.
#[command]
pub async fn pause_all_sources() -> Result<(), String> {
    log::info!("Pausing all sources");

    Ok(())

    // TODO: Wire to Python CLI
    // let args = vec!["sources", "pause-all"];
    // crate::python::execute_cli_void(&args).await?;
    // Ok(())
}

/// Resume all paused data sources.
#[command]
pub async fn resume_all_sources() -> Result<(), String> {
    log::info!("Resuming all sources");

    Ok(())

    // TODO: Wire to Python CLI
    // let args = vec!["sources", "resume-all"];
    // crate::python::execute_cli_void(&args).await?;
    // Ok(())
}
