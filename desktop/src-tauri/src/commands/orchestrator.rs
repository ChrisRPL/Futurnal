//! Orchestrator status IPC commands.
//!
//! Handles orchestrator daemon lifecycle and status monitoring.

use serde::{Deserialize, Serialize};
use tauri::command;

/// Orchestrator daemon status response.
///
/// Matches the JSON output from `futurnal orchestrator daemon-status --json`.
#[derive(Debug, Serialize, Deserialize)]
pub struct OrchestratorStatus {
    pub running: bool,
    pub pid: Option<u32>,
    pub workspace: String,
    pub stale_pid_file: bool,
}

/// Orchestrator status response with job queue metrics.
#[derive(Debug, Serialize, Deserialize)]
pub struct OrchestratorFullStatus {
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

/// Get workspace path: ~/.futurnal/workspace
fn get_workspace_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    format!("{home}/.futurnal/workspace")
}

/// Get the orchestrator daemon status.
///
/// Calls: `futurnal orchestrator daemon-status --workspace ~/.futurnal/workspace --json`
#[command]
pub async fn get_orchestrator_status() -> Result<OrchestratorStatus, String> {
    log::info!("Getting orchestrator daemon status");

    let workspace = get_workspace_path();
    let status: OrchestratorStatus =
        crate::python::execute_cli(&["orchestrator", "daemon-status", "--workspace", &workspace, "--json"]).await?;

    Ok(status)
}

/// Start the orchestrator daemon.
///
/// Calls: `futurnal orchestrator start --workspace ~/.futurnal/workspace`
///
/// Note: This spawns the orchestrator as a background daemon process.
/// The command returns immediately after spawning.
#[command]
pub async fn start_orchestrator() -> Result<(), String> {
    log::info!("Starting orchestrator daemon");

    let workspace = get_workspace_path();

    // Spawn in background - the CLI runs the daemon in foreground
    // so we need to spawn it detached
    crate::python::spawn_cli_background(&["orchestrator", "start", "--workspace", &workspace]).await?;

    // Give it a moment to initialize and write PID file
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    Ok(())
}

/// Stop the orchestrator daemon.
///
/// Calls: `futurnal orchestrator stop --workspace ~/.futurnal/workspace`
#[command]
pub async fn stop_orchestrator() -> Result<(), String> {
    log::info!("Stopping orchestrator daemon");

    let workspace = get_workspace_path();
    crate::python::execute_cli_void(&["orchestrator", "stop", "--workspace", &workspace]).await?;

    Ok(())
}

/// Ensure orchestrator is running, starting it if not.
///
/// Returns true if the orchestrator was started, false if already running.
#[command]
pub async fn ensure_orchestrator_running() -> Result<bool, String> {
    log::info!("Ensuring orchestrator is running");

    let workspace = get_workspace_path();

    // Check current status
    let status: OrchestratorStatus =
        crate::python::execute_cli(&["orchestrator", "daemon-status", "--workspace", &workspace, "--json"]).await?;

    if !status.running {
        log::info!("Orchestrator not running, starting...");
        // Spawn in background - returns immediately
        crate::python::spawn_cli_background(&["orchestrator", "start", "--workspace", &workspace]).await?;

        // Give it a moment to initialize
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        return Ok(true); // Started
    }

    log::info!("Orchestrator already running (PID: {:?})", status.pid);
    Ok(false) // Already running
}
