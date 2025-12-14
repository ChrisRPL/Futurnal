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

/// Find the Futurnal root directory by looking for .venv or pyproject.toml.
///
/// Searches upward from the current exe path to find the project root.
fn find_futurnal_root() -> Option<std::path::PathBuf> {
    // Try from current exe path (works in dev and release)
    if let Ok(exe_path) = std::env::current_exe() {
        let mut current = exe_path.parent().map(|p| p.to_path_buf());
        while let Some(dir) = current {
            // Check for .venv directory or pyproject.toml
            if dir.join(".venv").exists() || dir.join("pyproject.toml").exists() {
                return Some(dir);
            }
            current = dir.parent().map(|p| p.to_path_buf());
        }
    }

    // Fallback: check if we're already in a directory with .venv
    if let Ok(cwd) = std::env::current_dir() {
        if cwd.join(".venv").exists() {
            return Some(cwd);
        }
    }

    None
}

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
    // Find Futurnal root directory (where .venv is located)
    let futurnal_root = find_futurnal_root();

    // Use venv Python if available, otherwise fall back to system python3
    let python_path = futurnal_root
        .as_ref()
        .map(|root| root.join(".venv/bin/python3"))
        .filter(|p| p.exists())
        .unwrap_or_else(|| std::path::PathBuf::from("python3"));

    let mut cmd = Command::new(&python_path);
    cmd.arg("-m")
        .arg("futurnal.cli")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    // Set working directory to Futurnal root
    if let Some(root) = futurnal_root {
        cmd.current_dir(&root);
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

/// Execute a CLI command and return raw string output (for non-JSON commands).
///
/// # Arguments
/// * `args` - Command arguments
///
/// # Returns
/// Raw stdout string or error
pub async fn execute_cli_raw(args: &[&str]) -> Result<String, PythonError> {
    execute_cli_raw_with_timeout(args, DEFAULT_TIMEOUT_SECS).await
}

/// Execute a CLI command and return raw string output with custom timeout.
pub async fn execute_cli_raw_with_timeout(
    args: &[&str],
    timeout_secs: u64,
) -> Result<String, PythonError> {
    // Find Futurnal root directory (where .venv is located)
    let futurnal_root = find_futurnal_root();

    // Use venv Python if available, otherwise fall back to system python3
    let python_path = futurnal_root
        .as_ref()
        .map(|root| root.join(".venv/bin/python3"))
        .filter(|p| p.exists())
        .unwrap_or_else(|| std::path::PathBuf::from("python3"));

    let mut cmd = Command::new(&python_path);
    cmd.arg("-m")
        .arg("futurnal.cli")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    // Set working directory to Futurnal root
    if let Some(root) = futurnal_root {
        cmd.current_dir(&root);
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
    Ok(stdout.to_string())
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
    // Find Futurnal root directory (where .venv is located)
    let futurnal_root = find_futurnal_root();

    // Use venv Python if available, otherwise fall back to system python3
    let python_path = futurnal_root
        .as_ref()
        .map(|root| root.join(".venv/bin/python3"))
        .filter(|p| p.exists())
        .unwrap_or_else(|| std::path::PathBuf::from("python3"));

    let mut cmd = Command::new(&python_path);
    cmd.arg("-m")
        .arg("futurnal.cli")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    // Set working directory to Futurnal root
    if let Some(root) = futurnal_root {
        cmd.current_dir(&root);
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

/// Spawn a CLI command in the background (daemon mode).
///
/// This function spawns the process and returns immediately without waiting.
/// Used for long-running processes like the orchestrator daemon.
///
/// # Arguments
/// * `args` - Command arguments (e.g., ["orchestrator", "start"])
///
/// # Returns
/// Ok(()) if spawn succeeded, error otherwise
pub async fn spawn_cli_background(args: &[&str]) -> Result<(), PythonError> {
    use std::process::Command as StdCommand;

    let mut cmd = StdCommand::new("python3");
    cmd.arg("-m")
        .arg("futurnal.cli")
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .stdin(Stdio::null());

    // Set working directory to parent of desktop (where src/futurnal is)
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(parent) = exe_path.parent() {
            let futurnal_root = parent.parent().unwrap_or(parent);
            cmd.current_dir(futurnal_root);
        }
    }

    // Spawn detached process
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // Create new process group so it survives parent exit
        cmd.process_group(0);
    }

    cmd.spawn()?;

    log::info!("Spawned background process: python3 -m futurnal.cli {}", args.join(" "));

    Ok(())
}
