//! Ollama model management commands.
//!
//! Provides commands for listing installed models and pulling new ones.

use serde::{Deserialize, Serialize};
use std::process::Stdio;
use tauri::command;
use tauri::Emitter;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

/// Information about an installed Ollama model.
#[derive(Debug, Serialize, Deserialize)]
pub struct OllamaModel {
    pub name: String,
    pub size: String,
    pub modified: String,
}

/// Response from listing models.
#[derive(Debug, Serialize, Deserialize)]
pub struct ListModelsResponse {
    pub models: Vec<OllamaModel>,
}

/// Progress update during model pull.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullProgress {
    pub status: String,
    pub completed: u64,
    pub total: u64,
    pub percent: f64,
}

/// List installed Ollama models.
#[command]
pub async fn list_ollama_models() -> Result<ListModelsResponse, String> {
    let output = Command::new("ollama")
        .arg("list")
        .output()
        .await
        .map_err(|e| format!("Failed to run ollama list: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("ollama list failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut models = Vec::new();

    // Parse the output (skip header line)
    for line in stdout.lines().skip(1) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 4 {
            models.push(OllamaModel {
                name: parts[0].to_string(),
                size: parts[2].to_string(),
                modified: parts[3..].join(" "),
            });
        }
    }

    log::info!("Found {} installed Ollama models", models.len());
    Ok(ListModelsResponse { models })
}

/// Check if a specific model is installed.
#[command]
pub async fn is_model_installed(model_name: String) -> Result<bool, String> {
    let response = list_ollama_models().await?;

    // Check if model name matches (handle tag variations)
    let is_installed = response.models.iter().any(|m| {
        m.name == model_name ||
        m.name.starts_with(&format!("{}:", model_name)) ||
        model_name.starts_with(&format!("{}:", m.name.split(':').next().unwrap_or("")))
    });

    Ok(is_installed)
}

/// Pull (download) an Ollama model.
///
/// This command streams progress updates via Tauri events.
#[command]
pub async fn pull_ollama_model(
    app_handle: tauri::AppHandle,
    model_name: String,
) -> Result<bool, String> {
    log::info!("Starting pull for model: {}", model_name);

    let mut child = Command::new("ollama")
        .arg("pull")
        .arg(&model_name)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start ollama pull: {}", e))?;

    // Read stderr for progress (ollama outputs progress to stderr)
    if let Some(stderr) = child.stderr.take() {
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();

        while let Ok(Some(line)) = lines.next_line().await {
            // Parse progress from ollama output
            // Format: "pulling manifest" or "pulling sha256:..." or percentage
            let progress = parse_pull_progress(&line);

            // Emit progress event to frontend
            let _ = app_handle.emit("ollama-pull-progress", &serde_json::json!({
                "model": model_name,
                "status": progress.status,
                "completed": progress.completed,
                "total": progress.total,
                "percent": progress.percent,
            }));
        }
    }

    let status = child
        .wait()
        .await
        .map_err(|e| format!("Failed to wait for ollama pull: {}", e))?;

    if status.success() {
        log::info!("Successfully pulled model: {}", model_name);

        // Emit completion event
        let _ = app_handle.emit("ollama-pull-complete", &serde_json::json!({
            "model": model_name,
            "success": true,
        }));

        Ok(true)
    } else {
        log::error!("Failed to pull model: {}", model_name);

        // Emit failure event
        let _ = app_handle.emit("ollama-pull-complete", &serde_json::json!({
            "model": model_name,
            "success": false,
        }));

        Err(format!("Failed to pull model: {}", model_name))
    }
}

/// Parse progress from ollama pull output.
fn parse_pull_progress(line: &str) -> PullProgress {
    // Try to extract percentage if present
    if let Some(pct_str) = line.split('%').next() {
        if let Some(num_str) = pct_str.split_whitespace().last() {
            if let Ok(pct) = num_str.parse::<f64>() {
                return PullProgress {
                    status: line.to_string(),
                    completed: (pct * 100.0) as u64,
                    total: 10000,
                    percent: pct,
                };
            }
        }
    }

    // Default progress for status messages
    PullProgress {
        status: line.to_string(),
        completed: 0,
        total: 0,
        percent: 0.0,
    }
}
