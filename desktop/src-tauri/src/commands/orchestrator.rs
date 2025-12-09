//! Orchestrator status IPC commands.
//!
//! Handles orchestrator status and job queue monitoring.

use serde::{Deserialize, Serialize};
use tauri::command;

/// Orchestrator status response.
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

/// Status of an individual source within orchestrator.
#[derive(Debug, Serialize, Deserialize)]
pub struct SourceStatus {
    pub id: String,
    pub name: String,
    pub status: String,
    pub progress: Option<SyncProgress>,
    pub last_sync: Option<String>,
    pub error: Option<String>,
}

/// Sync progress information.
#[derive(Debug, Serialize, Deserialize)]
pub struct SyncProgress {
    pub current: u32,
    pub total: u32,
    pub phase: String,
}

/// Get the current orchestrator status.
///
/// Calls: `futurnal orchestrator status --format json`
#[command]
pub async fn get_orchestrator_status() -> Result<OrchestratorStatus, String> {
    log::info!("Getting orchestrator status");

    // Return default status - orchestrator not running until started
    let status = OrchestratorStatus {
        running: false,
        active_jobs: 0,
        pending_jobs: 0,
        failed_jobs: 0,
        sources: vec![],
        last_activity: None,
        uptime_seconds: 0,
    };

    Ok(status)

    // TODO: Wire to Python CLI
    // let args = vec!["orchestrator", "status", "--format", "json"];
    // let status: OrchestratorStatus = crate::python::execute_cli(&args).await?;
    // Ok(status)
}
