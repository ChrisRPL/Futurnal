//! Connector management IPC commands.
//!
//! Handles data source (connector) management operations.
//! Routes to appropriate Python CLI commands based on connector type.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
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

impl Default for ConnectorStats {
    fn default() -> Self {
        Self {
            files_processed: 0,
            entities_extracted: 0,
            last_duration: None,
        }
    }
}

/// Request to add a new data source.
#[derive(Debug, Deserialize)]
pub struct AddSourceRequest {
    pub connector_type: String,
    pub name: String,
    pub config: serde_json::Value,
}

/// Raw descriptor from Obsidian CLI.
#[derive(Debug, Deserialize)]
struct ObsidianDescriptor {
    id: String,
    name: Option<String>,
    base_path: String,
    #[serde(default)]
    paused: bool,
}

/// Raw descriptor from IMAP CLI.
#[derive(Debug, Deserialize)]
struct ImapDescriptor {
    id: String,
    name: Option<String>,
    email_address: String,
    imap_host: String,
    #[serde(default)]
    paused: bool,
}

/// Raw descriptor from GitHub CLI.
#[derive(Debug, Deserialize)]
struct GithubDescriptor {
    id: String,
    name: Option<String>,
    full_name: String,
    github_host: String,
    #[serde(default)]
    visibility: String,
}

/// Convert Obsidian descriptor to Connector.
fn obsidian_to_connector(desc: ObsidianDescriptor) -> Connector {
    Connector {
        id: desc.id.clone(),
        name: desc.name.unwrap_or_else(|| format!("Obsidian-{}", &desc.id[..8.min(desc.id.len())])),
        connector_type: "obsidian".to_string(),
        status: if desc.paused { "paused" } else { "active" }.to_string(),
        last_sync: None,
        next_sync: None,
        progress: None,
        error: None,
        config: serde_json::json!({
            "path": desc.base_path
        }),
        stats: ConnectorStats::default(),
    }
}

/// Convert IMAP descriptor to Connector.
fn imap_to_connector(desc: ImapDescriptor) -> Connector {
    Connector {
        id: desc.id.clone(),
        name: desc.name.unwrap_or_else(|| format!("IMAP-{}", &desc.id[..8.min(desc.id.len())])),
        connector_type: "imap".to_string(),
        status: if desc.paused { "paused" } else { "active" }.to_string(),
        last_sync: None,
        next_sync: None,
        progress: None,
        error: None,
        config: serde_json::json!({
            "server": desc.imap_host,
            "email": desc.email_address
        }),
        stats: ConnectorStats::default(),
    }
}

/// Convert GitHub descriptor to Connector.
fn github_to_connector(desc: GithubDescriptor) -> Connector {
    Connector {
        id: desc.id.clone(),
        name: desc.name.unwrap_or_else(|| desc.full_name.clone()),
        connector_type: "github".to_string(),
        status: "active".to_string(),
        last_sync: None,
        next_sync: None,
        progress: None,
        error: None,
        config: serde_json::json!({
            "repo": desc.full_name,
            "host": desc.github_host,
            "visibility": desc.visibility
        }),
        stats: ConnectorStats::default(),
    }
}

/// Raw descriptor from local sources CLI.
#[derive(Debug, Deserialize)]
struct LocalSourceDescriptor {
    id: String,
    name: String,
    connector_type: String,
    path: String,
}

/// Convert local source descriptor to Connector.
fn local_source_to_connector(desc: LocalSourceDescriptor) -> Connector {
    Connector {
        id: desc.id.clone(),
        name: desc.name,
        connector_type: desc.connector_type,
        status: "active".to_string(),
        last_sync: None,
        next_sync: None,
        progress: None,
        error: None,
        config: serde_json::json!({
            "path": desc.path
        }),
        stats: ConnectorStats::default(),
    }
}

/// Stats gathered from parsed files for a source.
#[derive(Debug, Default)]
struct SourceStats {
    files_processed: u32,
    last_sync: Option<String>,
}

/// Get workspace path: ~/.futurnal/workspace
fn get_workspace_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(format!("{home}/.futurnal/workspace"))
}

/// Gather stats from parsed files for all sources.
/// Reads parsed/*.json files and counts files per source.
fn gather_source_stats() -> HashMap<String, SourceStats> {
    let mut stats: HashMap<String, SourceStats> = HashMap::new();
    let parsed_dir = get_workspace_path().join("parsed");

    if !parsed_dir.exists() {
        return stats;
    }

    let entries = match std::fs::read_dir(&parsed_dir) {
        Ok(e) => e,
        Err(_) => return stats,
    };

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();
        if !path.extension().map_or(false, |ext| ext == "json") {
            continue;
        }

        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Parse the JSON to get source and ingested_at
        let parsed: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Get source name from metadata
        let source = parsed
            .get("metadata")
            .and_then(|m| m.get("source"))
            .and_then(|s| s.as_str())
            .unwrap_or("unknown");

        // Get ingested_at timestamp
        let ingested_at = parsed
            .get("metadata")
            .and_then(|m| m.get("ingested_at"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string());

        let entry = stats.entry(source.to_string()).or_default();
        entry.files_processed += 1;

        // Track most recent timestamp
        if let Some(timestamp) = ingested_at {
            if entry.last_sync.is_none() || entry.last_sync.as_ref().map_or(true, |t| &timestamp > t) {
                entry.last_sync = Some(timestamp);
            }
        }
    }

    stats
}

/// List all configured data sources.
///
/// Aggregates sources from:
/// - Local sources: `futurnal sources list --json`
/// - Obsidian vaults: `futurnal sources obsidian list --json`
/// - IMAP mailboxes: `futurnal sources imap list --json`
/// - GitHub repos: `futurnal github list --json`
#[command]
pub async fn list_sources() -> Result<Vec<Connector>, String> {
    let mut connectors: Vec<Connector> = Vec::new();

    // Gather stats from parsed files
    let source_stats = gather_source_stats();

    // List local/generic sources (includes local_folder type)
    match crate::python::execute_cli::<Vec<LocalSourceDescriptor>>(
        &["sources", "list", "--json"]
    ).await {
        Ok(sources) => {
            for s in sources {
                let mut connector = local_source_to_connector(s);
                // Apply stats from parsed files
                if let Some(stats) = source_stats.get(&connector.name) {
                    connector.last_sync = stats.last_sync.clone();
                    connector.stats.files_processed = stats.files_processed;
                    // Update status based on whether files have been processed
                    if stats.files_processed > 0 {
                        connector.status = "active".to_string();
                    }
                }
                connectors.push(connector);
            }
        }
        Err(e) => {
            log::warn!("Failed to list local sources: {}", e);
        }
    }

    // List Obsidian vaults
    match crate::python::execute_cli::<Vec<ObsidianDescriptor>>(
        &["sources", "obsidian", "list", "--json"]
    ).await {
        Ok(vaults) => {
            for v in vaults {
                let mut connector = obsidian_to_connector(v);
                // Apply stats from parsed files
                if let Some(stats) = source_stats.get(&connector.name) {
                    connector.last_sync = stats.last_sync.clone();
                    connector.stats.files_processed = stats.files_processed;
                }
                connectors.push(connector);
            }
        }
        Err(e) => {
            log::warn!("Failed to list Obsidian vaults: {}", e);
        }
    }

    // List IMAP mailboxes
    match crate::python::execute_cli::<Vec<ImapDescriptor>>(
        &["sources", "imap", "list", "--json"]
    ).await {
        Ok(mailboxes) => {
            for m in mailboxes {
                let mut connector = imap_to_connector(m);
                // Apply stats from parsed files
                if let Some(stats) = source_stats.get(&connector.name) {
                    connector.last_sync = stats.last_sync.clone();
                    connector.stats.files_processed = stats.files_processed;
                }
                connectors.push(connector);
            }
        }
        Err(e) => {
            log::warn!("Failed to list IMAP mailboxes: {}", e);
        }
    }

    // List GitHub repositories
    match crate::python::execute_cli::<Vec<GithubDescriptor>>(
        &["github", "list", "--json"]
    ).await {
        Ok(repos) => {
            for r in repos {
                let mut connector = github_to_connector(r);
                // Apply stats from parsed files
                if let Some(stats) = source_stats.get(&connector.name) {
                    connector.last_sync = stats.last_sync.clone();
                    connector.stats.files_processed = stats.files_processed;
                }
                connectors.push(connector);
            }
        }
        Err(e) => {
            log::warn!("Failed to list GitHub repos: {}", e);
        }
    }

    Ok(connectors)
}

/// Add a new data source.
///
/// Routes by type:
/// - obsidian: `sources obsidian add --path <path> --name <name> --json`
/// - local_folder: `sources register --name <name> --root <path>`
/// - imap: `sources imap add --email <email> --host <host> --name <name> --json`
/// - github: `github add <owner/repo> --auth token --token <pat> --name <name>`
///
/// After successfully adding a source, ensures the orchestrator daemon is running
/// so that files get processed automatically.
#[command]
pub async fn add_source(request: AddSourceRequest) -> Result<Connector, String> {
    log::info!(
        "Adding source: {} ({})",
        request.name,
        request.connector_type
    );

    let config = request.config.as_object()
        .ok_or("Invalid config: expected object")?;

    let connector = match request.connector_type.as_str() {
        "obsidian" => {
            let path = config.get("path")
                .and_then(|v| v.as_str())
                .ok_or("Missing 'path' in config")?;

            let result: ObsidianDescriptor = crate::python::execute_cli(&[
                "sources", "obsidian", "add",
                "--path", path,
                "--name", &request.name,
                "--json"
            ]).await?;

            obsidian_to_connector(result)
        }
        "local_folder" => {
            let path = config.get("path")
                .and_then(|v| v.as_str())
                .ok_or("Missing 'path' in config")?;

            // Default excludes for common hidden/system directories
            let default_excludes = ".obsidian/**,.trash/**,.DS_Store,**/.git/**,**/node_modules/**";

            // Local folder uses 'register' command with @interval schedule for auto-sync
            crate::python::execute_cli_void(&[
                "sources", "register",
                "--name", &request.name,
                "--root", path,
                "--exclude", default_excludes,
                "--schedule", "@interval",
                "--interval-seconds", "300"  // 5 minute sync interval
            ]).await?;

            // Return a connector with the info we have
            Connector {
                id: request.name.clone(),
                name: request.name,
                connector_type: "local_folder".to_string(),
                status: "active".to_string(),
                last_sync: None,
                next_sync: None,
                progress: None,
                error: None,
                config: serde_json::json!({ "path": path }),
                stats: ConnectorStats::default(),
            }
        }
        "imap" => {
            let email = config.get("email")
                .and_then(|v| v.as_str())
                .ok_or("Missing 'email' in config")?;
            let server = config.get("server")
                .and_then(|v| v.as_str())
                .ok_or("Missing 'server' in config")?;

            let result: ImapDescriptor = crate::python::execute_cli(&[
                "sources", "imap", "add",
                "--email", email,
                "--host", server,
                "--name", &request.name,
                "--json"
            ]).await?;

            imap_to_connector(result)
        }
        "github" => {
            let repo = config.get("repo")
                .and_then(|v| v.as_str())
                .ok_or("Missing 'repo' in config")?;
            let token = config.get("token")
                .and_then(|v| v.as_str());

            let mut args = vec![
                "github", "add", repo,
                "--name", &request.name,
            ];

            // Add token auth if provided
            if let Some(t) = token {
                if !t.is_empty() {
                    args.extend(&["--auth", "token", "--token", t]);
                }
            }

            // GitHub add doesn't return JSON, so we construct the connector
            crate::python::execute_cli_void(&args).await?;

            Connector {
                id: repo.replace("/", "_"),
                name: request.name,
                connector_type: "github".to_string(),
                status: "active".to_string(),
                last_sync: None,
                next_sync: None,
                progress: None,
                error: None,
                config: serde_json::json!({ "repo": repo }),
                stats: ConnectorStats::default(),
            }
        }
        other => {
            return Err(format!("Unsupported connector type: {}", other));
        }
    };

    // Auto-start orchestrator so files get processed automatically
    // We don't fail the add_source if orchestrator fails to start
    match super::orchestrator::ensure_orchestrator_running().await {
        Ok(started) => {
            if started {
                log::info!("Orchestrator started automatically after adding source");
            } else {
                log::info!("Orchestrator already running");
            }
        }
        Err(e) => {
            log::warn!("Failed to auto-start orchestrator: {}. Files may not be processed until orchestrator is started manually.", e);
        }
    }

    Ok(connector)
}

/// Pause a data source.
///
/// Calls: `futurnal sources pause <name>`
#[command]
pub async fn pause_source(id: String) -> Result<(), String> {
    log::info!("Pausing source: {}", id);

    crate::python::execute_cli_void(&["sources", "pause", &id]).await?;
    Ok(())
}

/// Resume a paused data source.
///
/// Calls: `futurnal sources resume <name>`
#[command]
pub async fn resume_source(id: String) -> Result<(), String> {
    log::info!("Resuming source: {}", id);

    crate::python::execute_cli_void(&["sources", "resume", &id]).await?;
    Ok(())
}

/// Delete a data source.
///
/// Routes by ID pattern or tries multiple remove commands:
/// - Obsidian: `sources obsidian remove <id> --yes`
/// - IMAP: `sources imap remove <id> --yes`
/// - GitHub: `github remove <id> --yes`
/// - Generic: `sources remove <id> --yes`
#[command]
pub async fn delete_source(id: String) -> Result<(), String> {
    log::info!("Deleting source: {}", id);

    // Try generic remove first (handles both local sources and can detect type)
    match crate::python::execute_cli_void(&["sources", "remove", &id, "--yes"]).await {
        Ok(()) => return Ok(()),
        Err(_) => {}
    }

    // Try Obsidian-specific remove
    match crate::python::execute_cli_void(&["sources", "obsidian", "remove", &id, "--yes"]).await {
        Ok(()) => return Ok(()),
        Err(_) => {}
    }

    // Try IMAP-specific remove
    match crate::python::execute_cli_void(&["sources", "imap", "remove", &id, "--yes"]).await {
        Ok(()) => return Ok(()),
        Err(_) => {}
    }

    // Try GitHub-specific remove
    match crate::python::execute_cli_void(&["github", "remove", &id, "--yes"]).await {
        Ok(()) => return Ok(()),
        Err(e) => {
            Err(format!("Failed to remove source '{}': {}", id, e))
        }
    }
}

/// Retry a failed source sync.
///
/// Calls: `futurnal sources quarantine retry <id> --source-name <id>`
#[command]
pub async fn retry_source(id: String) -> Result<(), String> {
    log::info!("Retrying source: {}", id);

    crate::python::execute_cli_void(&[
        "sources", "quarantine", "retry", &id,
        "--source-name", &id
    ]).await?;
    Ok(())
}

/// Pause all data sources.
///
/// Iterates all active sources and pauses each one.
#[command]
pub async fn pause_all_sources() -> Result<(), String> {
    log::info!("Pausing all sources");

    let connectors = list_sources().await?;
    let mut errors = Vec::new();

    for connector in connectors.iter().filter(|c| c.status == "active" || c.status == "syncing") {
        if let Err(e) = pause_source(connector.id.clone()).await {
            log::warn!("Failed to pause {}: {}", connector.id, e);
            errors.push(format!("{}: {}", connector.id, e));
        }
    }

    if !errors.is_empty() {
        Err(format!("Some sources failed to pause: {}", errors.join(", ")))
    } else {
        Ok(())
    }
}

/// Resume all paused data sources.
///
/// Iterates all paused sources and resumes each one.
#[command]
pub async fn resume_all_sources() -> Result<(), String> {
    log::info!("Resuming all sources");

    let connectors = list_sources().await?;
    let mut errors = Vec::new();

    for connector in connectors.iter().filter(|c| c.status == "paused") {
        if let Err(e) = resume_source(connector.id.clone()).await {
            log::warn!("Failed to resume {}: {}", connector.id, e);
            errors.push(format!("{}: {}", connector.id, e));
        }
    }

    if !errors.is_empty() {
        Err(format!("Some sources failed to resume: {}", errors.join(", ")))
    } else {
        Ok(())
    }
}
