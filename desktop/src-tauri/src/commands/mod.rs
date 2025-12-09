//! Tauri IPC command handlers.
//!
//! All commands in this module bridge the React frontend to the Python backend
//! via the futurnal CLI.

pub mod connectors;
pub mod graph;
pub mod orchestrator;
pub mod privacy;
pub mod search;

pub use connectors::*;
pub use graph::*;
pub use orchestrator::*;
pub use privacy::*;
pub use search::*;
