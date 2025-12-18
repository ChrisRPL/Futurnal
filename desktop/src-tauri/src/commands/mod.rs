//! Tauri IPC command handlers.
//!
//! All commands in this module bridge the React frontend to the Python backend
//! via the futurnal CLI.

pub mod activity;
pub mod causal;
pub mod chat;
pub mod cloud_sync;
pub mod connectors;
pub mod graph;
pub mod insights;
pub mod learning;
pub mod multimodal;
pub mod ollama;
pub mod orchestrator;
pub mod papers;
pub mod privacy;
pub mod schema;
pub mod search;

pub use activity::*;
pub use causal::*;
pub use chat::*;
pub use cloud_sync::*;
pub use connectors::*;
pub use graph::*;
pub use insights::*;
pub use learning::*;
pub use multimodal::*;
pub use ollama::*;
pub use orchestrator::*;
pub use papers::*;
pub use privacy::*;
pub use schema::*;
pub use search::*;
