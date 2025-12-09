//! Knowledge graph IPC commands.
//!
//! Handles PKG visualization data retrieval.

use serde::{Deserialize, Serialize};
use tauri::command;

/// Graph node for visualization.
#[derive(Debug, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub timestamp: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

/// Graph edge/link for visualization.
#[derive(Debug, Serialize, Deserialize)]
pub struct GraphLink {
    pub source: String,
    pub target: String,
    pub relationship: String,
    pub weight: Option<f64>,
}

/// Complete graph data for react-force-graph.
#[derive(Debug, Serialize, Deserialize)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    pub links: Vec<GraphLink>,
}

/// Get knowledge graph data for visualization.
///
/// Calls: `futurnal graph export --json --limit <n>`
#[command]
pub async fn get_knowledge_graph(limit: Option<u32>) -> Result<GraphData, String> {
    let limit = limit.unwrap_or(1000);
    log::info!("Getting knowledge graph (limit: {})", limit);

    // Return empty graph - no data until ingestion runs
    let graph = GraphData {
        nodes: vec![],
        links: vec![],
    };

    Ok(graph)

    // TODO: Wire to Python CLI when graph export command exists
    // let limit_str = limit.to_string();
    // let args = vec!["graph", "export", "--json", "--limit", &limit_str];
    // let graph: GraphData = crate::python::execute_cli(&args).await?;
    // Ok(graph)
}
