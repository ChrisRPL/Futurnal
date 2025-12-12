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

/// GitHub sync status information.
struct GithubSyncStatus {
    is_synced: bool,
    files_count: u32,
    last_sync: Option<String>,
}

/// Recursively count files in a directory, excluding .git.
fn count_files_recursive(dir: &std::path::Path) -> u32 {
    let mut count = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            // Skip .git directory
            if path.file_name().map_or(false, |n| n == ".git") {
                continue;
            }
            if path.is_file() {
                count += 1;
            } else if path.is_dir() {
                count += count_files_recursive(&path);
            }
        }
    }
    count
}

/// Check if a GitHub repo has been synced (cloned) and get stats.
fn get_github_sync_status(repo_id: &str) -> GithubSyncStatus {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let clone_dir = PathBuf::from(format!("{home}/.futurnal/repositories/github/{repo_id}"));

    if !clone_dir.exists() {
        return GithubSyncStatus {
            is_synced: false,
            files_count: 0,
            last_sync: None,
        };
    }

    // Count files in the clone directory (excluding .git)
    let file_count = count_files_recursive(&clone_dir);

    // Get last modification time of the directory
    let last_sync = std::fs::metadata(&clone_dir)
        .and_then(|m| m.modified())
        .ok()
        .map(|t| {
            // Convert SystemTime to RFC3339 string
            let duration = t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
            let secs = duration.as_secs() as i64;
            // Simple ISO 8601 format without external deps
            format!("{}", chrono::DateTime::from_timestamp(secs, 0)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_else(|| "unknown".to_string()))
        });

    GithubSyncStatus {
        is_synced: true,
        files_count: file_count,
        last_sync,
    }
}

/// Convert GitHub descriptor to Connector.
fn github_to_connector(desc: GithubDescriptor) -> Connector {
    let sync_status = get_github_sync_status(&desc.id);

    Connector {
        id: desc.id.clone(),
        name: desc.name.unwrap_or_else(|| desc.full_name.clone()),
        connector_type: "github".to_string(),
        status: if sync_status.is_synced { "active" } else { "pending" }.to_string(),
        last_sync: sync_status.last_sync,
        next_sync: None,
        progress: None,
        error: if !sync_status.is_synced {
            Some("Not synced - click Sync to clone repository".to_string())
        } else {
            None
        },
        config: serde_json::json!({
            "repo": desc.full_name,
            "host": desc.github_host,
            "visibility": desc.visibility,
            "synced": sync_status.is_synced
        }),
        stats: ConnectorStats {
            files_processed: sync_status.files_count,
            entities_extracted: 0,
            last_duration: None,
        },
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
            let password = config.get("password")
                .and_then(|v| v.as_str());

            // Build command args
            let mut args = vec![
                "sources", "imap", "add",
                "--email", email,
                "--host", server,
                "--name", &request.name,
                "--auth", "app_password",  // Force app_password mode for desktop UI
                "--json"
            ];

            // Add password if provided
            let password_owned: String;
            if let Some(pwd) = password {
                if !pwd.is_empty() {
                    password_owned = pwd.to_string();
                    args.push("--password");
                    args.push(&password_owned);
                }
            }

            let result: ImapDescriptor = crate::python::execute_cli(&args).await?;

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
/// Also removes parsed files from the knowledge graph.
#[command]
pub async fn delete_source(id: String, connector_type: Option<String>) -> Result<(), String> {
    log::info!("Deleting source: {} (type: {:?})", id, connector_type);

    // First, clean up parsed files from the knowledge graph
    let cleanup_result = cleanup_parsed_files(&id, connector_type.as_deref()).await;
    if let Err(e) = &cleanup_result {
        log::warn!("Failed to cleanup parsed files for source '{}': {}", id, e);
    }

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

/// Clean up parsed files from the knowledge graph for a given source.
/// Scans parsed, imap, and entities directories and removes files matching the source pattern.
async fn cleanup_parsed_files(id: &str, connector_type: Option<&str>) -> Result<u32, String> {
    use std::fs;

    let home = dirs::home_dir().ok_or("Failed to get home directory")?;
    let workspace = home.join(".futurnal").join("workspace");
    let parsed_dir = workspace.join("parsed");
    let imap_dir = workspace.join("imap");
    let entities_dir = workspace.join("entities");

    // Build possible source_id patterns to match
    let mut patterns: Vec<String> = vec![
        id.to_string(),  // Direct ID match
    ];

    // Add type-specific patterns
    match connector_type {
        Some("github") => {
            patterns.push(format!("github-{}", id.replace('/', "-")));
        }
        Some("imap") => {
            patterns.push(format!("imap-{}", id));
        }
        Some("obsidian") => {
            patterns.push(format!("obsidian-{}", id));
        }
        Some("local_folder") => {
            // Local folder sources use the name as source
            patterns.push(id.to_string());
        }
        _ => {
            // For unknown types, try common patterns
            patterns.push(format!("github-{}", id.replace('/', "-")));
            patterns.push(format!("imap-{}", id));
            patterns.push(format!("obsidian-{}", id));
        }
    }

    log::info!("Cleaning up files matching patterns: {:?}", patterns);

    let mut removed_count = 0u32;

    // Helper closure to check if a JSON document matches our patterns
    let matches_pattern = |doc: &serde_json::Value| -> bool {
        // Check metadata.source_id
        if let Some(metadata) = doc.get("metadata") {
            if let Some(source_id) = metadata.get("source_id").and_then(|v| v.as_str()) {
                if patterns.iter().any(|p| source_id == p || source_id.contains(p)) {
                    return true;
                }
            }
            // Check metadata.source
            if let Some(source) = metadata.get("source").and_then(|v| v.as_str()) {
                if patterns.iter().any(|p| source == p || source.contains(p)) {
                    return true;
                }
            }
        }
        // Check top-level source (IMAP format)
        if let Some(source) = doc.get("source").and_then(|v| v.as_str()) {
            if patterns.iter().any(|p| source == p || source.contains(p)) {
                return true;
            }
        }
        false
    };

    // Helper to scan a directory and remove matching files
    let scan_and_remove = |dir: &std::path::Path, removed: &mut u32| {
        if !dir.exists() {
            return;
        }
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                if let Ok(content) = fs::read_to_string(&path) {
                    if let Ok(doc) = serde_json::from_str::<serde_json::Value>(&content) {
                        if matches_pattern(&doc) {
                            if fs::remove_file(&path).is_ok() {
                                *removed += 1;
                                log::debug!("Removed file: {:?}", path);
                            }
                        }
                    }
                }
            }
        }
    };

    // Scan all relevant directories
    scan_and_remove(&parsed_dir, &mut removed_count);
    scan_and_remove(&imap_dir, &mut removed_count);
    scan_and_remove(&entities_dir, &mut removed_count);

    log::info!("Removed {} files for source '{}'", removed_count, id);
    Ok(removed_count)
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

/// Result of a sync operation.
#[derive(Debug, Serialize, Deserialize)]
pub struct SyncResult {
    pub repo_id: String,
    pub full_name: String,
    pub status: String,
    pub files_synced: u32,
    pub bytes_synced: u64,
    pub bytes_synced_mb: f64,
    pub duration_seconds: f64,
    pub branches_synced: Vec<String>,
    pub error_message: Option<String>,
    /// Number of files processed into the knowledge graph (when --process flag used)
    pub files_processed: Option<u32>,
    /// Number of files that failed processing (when --process flag used)
    pub files_failed: Option<u32>,
}

/// Sync a data source (clone/update repository or scan local folder).
///
/// Routes by type:
/// - github: `github sync <id> --process --json` - clones/pulls repository files
/// - imap: `sources imap sync <id> --process --json` - syncs emails from IMAP server
/// - local_folder/obsidian: Enqueues job and processes via orchestrator
#[command]
pub async fn sync_source(id: String, connector_type: String) -> Result<SyncResult, String> {
    log::info!("Syncing source: {} (type: {})", id, connector_type);
    let start = std::time::Instant::now();

    match connector_type.as_str() {
        "github" => {
            // Call GitHub sync CLI to clone/update the repository and process files
            // The --process flag processes cloned files into the knowledge graph
            let result: SyncResult = crate::python::execute_cli_with_timeout(
                &["github", "sync", &id, "--process", "--json"],
                300  // 5 minute timeout for sync
            ).await?;

            if result.status == "completed" {
                log::info!(
                    "GitHub sync completed: {} files ({:.2} MB) in {:.1}s, processed: {:?}, failed: {:?}",
                    result.files_synced,
                    result.bytes_synced_mb,
                    result.duration_seconds,
                    result.files_processed,
                    result.files_failed
                );
            }

            Ok(result)
        }
        "imap" => {
            // Call IMAP sync CLI to sync emails and process them into the knowledge graph
            // Fast mode (default) uses batch IMAP fetch and simple text extraction for speed
            // The --limit flag limits initial sync to 50 emails to prevent long initial syncs
            let result: SyncResult = crate::python::execute_cli_with_timeout(
                &["sources", "imap", "sync", &id, "--limit", "50", "--json"],
                120  // 2 minute timeout for fast IMAP sync (~50 emails in ~10 seconds)
            ).await?;

            if result.status == "completed" || result.status == "completed_with_errors" {
                log::info!(
                    "IMAP sync completed: {} messages in {:.1}s, processed: {:?}, failed: {:?}",
                    result.files_synced,
                    result.duration_seconds,
                    result.files_processed,
                    result.files_failed
                );
            }

            Ok(result)
        }
        "local_folder" | "obsidian" => {
            // For local folder/obsidian sources, use synchronous sync command
            // This directly processes files without needing the orchestrator daemon
            // Use --force to ensure files are reprocessed even if already in state store
            log::info!("Starting synchronous sync for local source: {}", id);

            let result: SyncResult = crate::python::execute_cli_with_timeout(
                &["sources", "sync", &id, "--force", "--json"],
                300  // 5 minute timeout for sync
            ).await?;

            if result.status == "completed" || result.status == "completed_with_errors" {
                log::info!(
                    "Local folder sync completed: {} elements in {:.1}s",
                    result.files_synced,
                    result.duration_seconds
                );
            }

            Ok(result)
        }
        "local_folder_async" => {
            // Legacy async mode - enqueues job for orchestrator (kept for reference)
            if let Err(e) = super::orchestrator::ensure_orchestrator_running().await {
                log::warn!("Failed to ensure orchestrator is running: {}", e);
            }

            let enqueue_result = crate::python::execute_cli_void(&[
                "sources", "run", &id, "--force"
            ]).await;

            let duration = start.elapsed().as_secs_f64();

            match enqueue_result {
                Ok(()) => {
                    log::info!("Enqueued sync job for local source: {}", id);
                    Ok(SyncResult {
                        repo_id: id.clone(),
                        full_name: id,
                        status: "enqueued".to_string(),
                        files_synced: 0,
                        bytes_synced: 0,
                        bytes_synced_mb: 0.0,
                        duration_seconds: duration,
                        branches_synced: vec![],
                        error_message: Some("Job enqueued for processing.".to_string()),
                        files_processed: None,
                        files_failed: None,
                    })
                }
                Err(e) => {
                    log::error!("Failed to enqueue sync job for {}: {}", id, e);
                    Err(format!("Failed to sync: {}", e))
                }
            }
        }
        _ => {
            // For unknown types, try the generic run command
            if let Err(e) = super::orchestrator::ensure_orchestrator_running().await {
                log::warn!("Failed to ensure orchestrator is running: {}", e);
            }

            let enqueue_result = crate::python::execute_cli_void(&[
                "sources", "run", &id, "--force"
            ]).await;

            let duration = start.elapsed().as_secs_f64();

            match enqueue_result {
                Ok(()) => {
                    log::info!("Enqueued sync job for source: {}", id);
                    Ok(SyncResult {
                        repo_id: id.clone(),
                        full_name: id,
                        status: "enqueued".to_string(),
                        files_synced: 0,
                        bytes_synced: 0,
                        bytes_synced_mb: 0.0,
                        duration_seconds: duration,
                        branches_synced: vec![],
                        error_message: Some("Job enqueued for processing.".to_string()),
                        files_processed: None,
                        files_failed: None,
                    })
                }
                Err(e) => {
                    Err(format!(
                        "Sync for type '{}' failed. Make sure the orchestrator is running. Error: {}",
                        connector_type, e
                    ))
                }
            }
        }
    }
}

/// Sync all GitHub sources.
///
/// Calls: `github sync all --process --json`
#[command]
pub async fn sync_all_github() -> Result<Vec<SyncResult>, String> {
    log::info!("Syncing all GitHub repositories");

    // Use --process flag to process files into the knowledge graph
    let result: Vec<SyncResult> = crate::python::execute_cli_with_timeout(
        &["github", "sync", "all", "--process", "--json"],
        600  // 10 minute timeout for all syncs
    ).await?;

    Ok(result)
}

/// OAuth authentication result.
#[derive(Debug, Serialize, Deserialize)]
pub struct AuthResult {
    pub success: bool,
    pub mailbox_id: Option<String>,
    pub email: Option<String>,
    pub credential_id: Option<String>,
    pub provider: Option<String>,
    pub error: Option<String>,
}

/// Authenticate an IMAP mailbox using OAuth2.
///
/// This command opens a browser for OAuth2 authentication and stores the tokens.
/// Must be called after adding an IMAP source with `add_source`.
///
/// Calls: `sources imap authenticate <mailbox_id> --client-id <id> --client-secret <secret> --provider <provider> --json`
#[command]
pub async fn authenticate_imap(
    mailbox_id: String,
    client_id: String,
    client_secret: String,
    provider: Option<String>,
) -> Result<AuthResult, String> {
    let provider_name = provider.unwrap_or_else(|| "gmail".to_string());

    log::info!(
        "Authenticating IMAP mailbox: {} with provider: {}",
        mailbox_id,
        provider_name
    );

    let result: AuthResult = crate::python::execute_cli_with_timeout(
        &[
            "sources", "imap", "authenticate",
            &mailbox_id,
            "--client-id", &client_id,
            "--client-secret", &client_secret,
            "--provider", &provider_name,
            "--json"
        ],
        120  // 2 minute timeout for OAuth flow
    ).await?;

    if result.success {
        log::info!("IMAP authentication successful for: {}", mailbox_id);
    } else {
        log::warn!(
            "IMAP authentication failed for {}: {:?}",
            mailbox_id,
            result.error
        );
    }

    Ok(result)
}

/// Result of an IMAP connection test.
#[derive(Debug, Serialize, Deserialize)]
pub struct ConnectionTestResult {
    pub success: bool,
    pub message: Option<String>,
    pub error: Option<String>,
    pub folders: Option<u32>,
}

/// Test IMAP connection before saving.
///
/// Validates that the provided credentials can successfully connect to the IMAP server.
/// This should be called before `add_source` to give users immediate feedback on credential issues.
///
/// Calls: `sources imap test-connection --email <email> --host <host> --password <password> --json`
#[command]
pub async fn test_imap_connection(
    email: String,
    server: String,
    password: String,
) -> Result<ConnectionTestResult, String> {
    log::info!("Testing IMAP connection for: {}", email);

    let result: ConnectionTestResult = crate::python::execute_cli_with_timeout(
        &[
            "sources", "imap", "test-connection",
            "--email", &email,
            "--host", &server,
            "--password", &password,
            "--json"
        ],
        30  // 30 second timeout for connection test
    ).await?;

    if result.success {
        log::info!("IMAP connection test successful for: {}", email);
    } else {
        log::warn!(
            "IMAP connection test failed for {}: {:?}",
            email,
            result.error
        );
    }

    Ok(result)
}
