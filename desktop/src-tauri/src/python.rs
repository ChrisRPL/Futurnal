//! Python subprocess execution helper.
//!
//! Executes futurnal CLI commands and parses JSON output.
//! All IPC commands ultimately call the Python backend via this module.

use serde::de::DeserializeOwned;
use std::process::Stdio;
use thiserror::Error;
use tokio::process::Command;

/// Errors that can occur when executing Python commands.
#[derive(Error, Debug)]
pub enum PythonError {
    #[error("Failed to spawn Python process: {0}")]
    SpawnError(#[from] std::io::Error),

    #[error("Python command failed with exit code {exit_code:?}: {stderr}")]
    CommandFailed {
        stderr: String,
        exit_code: Option<i32>,
    },

    #[error("Command timed out after {timeout_secs} seconds")]
    Timeout { timeout_secs: u64 },

    #[error("Failed to parse JSON output: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("Python not found in PATH")]
    PythonNotFound,
}

impl From<PythonError> for String {
    fn from(err: PythonError) -> String {
        err.to_string()
    }
}

/// Default timeout for CLI commands in seconds.
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Execute a futurnal CLI command with JSON output.
///
/// # Arguments
/// * `args` - Command arguments (e.g., ["search", "--json", "query text"])
///
/// # Returns
/// Parsed JSON value or error
pub async fn execute_cli<T: DeserializeOwned>(args: &[&str]) -> Result<T, PythonError> {
    execute_cli_with_timeout(args, DEFAULT_TIMEOUT_SECS).await
}

/// Execute a futurnal CLI command with JSON output and custom timeout.
///
/// # Arguments
/// * `args` - Command arguments (e.g., ["search", "--json", "query text"])
/// * `timeout_secs` - Command timeout in seconds
///
/// # Returns
/// Parsed JSON value or error
pub async fn execute_cli_with_timeout<T: DeserializeOwned>(
    args: &[&str],
    timeout_secs: u64,
) -> Result<T, PythonError> {
    let mut cmd = Command::new("python3");
    cmd.arg("-m")
        .arg("futurnal.cli")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    // Set working directory to parent of desktop (where src/futurnal is)
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(parent) = exe_path.parent() {
            // In dev mode, we're in src-tauri, go up to Futurnal root
            let futurnal_root = parent.parent().unwrap_or(parent);
            cmd.current_dir(futurnal_root);
        }
    }

    let output = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        cmd.output(),
    )
    .await
    .map_err(|_| PythonError::Timeout { timeout_secs })??;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(PythonError::CommandFailed {
            stderr: stderr.to_string(),
            exit_code: output.status.code(),
        });
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let result: T = serde_json::from_str(&stdout)?;
    Ok(result)
}

/// Execute a CLI command that returns empty success (void operations).
pub async fn execute_cli_void(args: &[&str]) -> Result<(), PythonError> {
    execute_cli_void_with_timeout(args, DEFAULT_TIMEOUT_SECS).await
}

/// Execute a CLI command that returns empty success with custom timeout.
pub async fn execute_cli_void_with_timeout(
    args: &[&str],
    timeout_secs: u64,
) -> Result<(), PythonError> {
    let mut cmd = Command::new("python3");
    cmd.arg("-m")
        .arg("futurnal.cli")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    // Set working directory to parent of desktop (where src/futurnal is)
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(parent) = exe_path.parent() {
            let futurnal_root = parent.parent().unwrap_or(parent);
            cmd.current_dir(futurnal_root);
        }
    }

    let output = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        cmd.output(),
    )
    .await
    .map_err(|_| PythonError::Timeout { timeout_secs })??;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(PythonError::CommandFailed {
            stderr: stderr.to_string(),
            exit_code: output.status.code(),
        });
    }

    Ok(())
}

/// Check if Python and futurnal CLI are available.
pub async fn check_python_available() -> Result<bool, PythonError> {
    let mut cmd = Command::new("python3");
    cmd.arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    match cmd.output().await {
        Ok(output) => Ok(output.status.success()),
        Err(_) => Ok(false),
    }
}
