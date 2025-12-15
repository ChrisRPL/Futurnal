//! Tauri IPC command handlers.
//!
//! All commands in this module bridge the React frontend to the Python backend
//! via the futurnal CLI.

pub mod chat;
pub mod cloud_sync;
pub mod connectors;
pub mod graph;
pub mod ollama;
pub mod orchestrator;
pub mod privacy;
pub mod search;

pub use chat::*;
pub use cloud_sync::*;
pub use connectors::*;
pub use graph::*;
pub use ollama::*;
pub use orchestrator::*;
pub use privacy::*;
pub use search::*;
