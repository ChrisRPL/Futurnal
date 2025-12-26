//! Infrastructure management IPC commands.
//!
//! Handles auto-start and lifecycle management of Futurnal services:
//! - Neo4j database (via Docker)
//! - Orchestrator daemon
//!
//! Called on app startup to ensure all services are running.

use serde::{Deserialize, Serialize};
use tauri::command;

/// Service status information.
#[derive(Debug, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub running: bool,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub available: Option<bool>,
}

/// Docker service status.
#[derive(Debug, Serialize, Deserialize)]
pub struct DockerStatus {
    pub available: bool,
    pub status: String,
}

/// Complete infrastructure status.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InfrastructureStatus {
    pub success: bool,
    pub services: InfrastructureServices,
    pub all_healthy: bool,
}

/// Individual service statuses.
#[derive(Debug, Serialize, Deserialize)]
pub struct InfrastructureServices {
    pub docker: DockerStatus,
    pub neo4j: ServiceStatus,
    pub orchestrator: ServiceStatus,
}

/// Result of starting infrastructure.
#[derive(Debug, Serialize, Deserialize)]
pub struct ServiceStartResult {
    #[serde(default)]
    pub started: bool,
    #[serde(default, rename = "alreadyRunning")]
    pub already_running: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub available: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ready: Option<bool>,
}

/// Infrastructure start response from CLI.
#[derive(Debug, Serialize, Deserialize)]
pub struct InfrastructureStartResponse {
    pub success: bool,
    pub services: std::collections::HashMap<String, ServiceStartResult>,
    #[serde(default)]
    pub errors: Vec<String>,
}

/// Get infrastructure status.
///
/// Calls: `futurnal infrastructure status --json`
#[command]
pub async fn get_infrastructure_status() -> Result<InfrastructureStatus, String> {
    log::info!("Getting infrastructure status");

    let status: InfrastructureStatus =
        crate::python::execute_cli(&["infrastructure", "status", "--json"]).await?;

    Ok(status)
}

/// Start all infrastructure services.
///
/// Calls: `futurnal infrastructure start --json`
///
/// This starts:
/// - Neo4j Docker container (creates if needed)
/// - Orchestrator daemon
#[command]
pub async fn start_infrastructure() -> Result<InfrastructureStartResponse, String> {
    log::info!("Starting infrastructure services");

    // Use longer timeout as Neo4j startup can take time
    let result: InfrastructureStartResponse =
        crate::python::execute_cli_with_timeout(&["infrastructure", "start", "--json"], 120).await?;

    if result.success {
        log::info!("Infrastructure started successfully");
    } else {
        log::warn!("Infrastructure start completed with errors: {:?}", result.errors);
    }

    Ok(result)
}

/// Stop all infrastructure services.
///
/// Calls: `futurnal infrastructure stop --json`
#[command]
pub async fn stop_infrastructure() -> Result<serde_json::Value, String> {
    log::info!("Stopping infrastructure services");

    let result: serde_json::Value =
        crate::python::execute_cli(&["infrastructure", "stop", "--json"]).await?;

    Ok(result)
}

/// Ensure all infrastructure is running, starting if needed.
///
/// Returns the current status after ensuring services are up.
/// This is the main entry point called on app startup.
#[command]
pub async fn ensure_infrastructure_running() -> Result<InfrastructureStatus, String> {
    log::info!("Ensuring infrastructure is running");

    // First check current status
    let status: InfrastructureStatus =
        crate::python::execute_cli(&["infrastructure", "status", "--json"]).await?;

    if status.all_healthy {
        log::info!("All infrastructure already healthy");
        return Ok(status);
    }

    // Start infrastructure if not healthy
    log::info!("Infrastructure not fully healthy, starting services...");
    let _start_result: InfrastructureStartResponse =
        crate::python::execute_cli_with_timeout(&["infrastructure", "start", "--json"], 120).await?;

    // Return updated status
    let new_status: InfrastructureStatus =
        crate::python::execute_cli(&["infrastructure", "status", "--json"]).await?;

    Ok(new_status)
}
