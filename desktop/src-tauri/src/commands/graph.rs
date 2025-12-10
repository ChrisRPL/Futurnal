//! Knowledge graph IPC commands.
//!
//! Handles PKG visualization data retrieval.
//! Reads parsed documents from workspace and converts to graph format.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
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

/// Parsed element from Unstructured.io output.
#[derive(Debug, Deserialize)]
struct ParsedElement {
    #[serde(rename = "type")]
    element_type: String,
    element_id: String,
    text: String,
    metadata: ParsedMetadata,
}

/// Metadata from parsed element.
#[derive(Debug, Deserialize)]
struct ParsedMetadata {
    filename: Option<String>,
    source: Option<String>,
    ingested_at: Option<String>,
    #[serde(flatten)]
    extra: HashMap<String, serde_json::Value>,
}

/// Get workspace path: ~/.futurnal/workspace
fn get_workspace_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(format!("{home}/.futurnal/workspace"))
}

/// Get knowledge graph data for visualization.
///
/// Reads parsed documents from workspace/parsed/ directory
/// and converts them to graph nodes.
#[command]
pub async fn get_knowledge_graph(limit: Option<u32>) -> Result<GraphData, String> {
    let limit = limit.unwrap_or(1000) as usize;
    log::info!("Getting knowledge graph (limit: {})", limit);

    let parsed_dir = get_workspace_path().join("parsed");

    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut links: Vec<GraphLink> = Vec::new();
    let mut source_nodes: HashMap<String, String> = HashMap::new();

    // Read parsed files
    if parsed_dir.exists() {
        let entries = std::fs::read_dir(&parsed_dir)
            .map_err(|e| format!("Failed to read parsed directory: {}", e))?;

        for entry in entries.take(limit) {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "json") {
                let content = match std::fs::read_to_string(&path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                let element: ParsedElement = match serde_json::from_str(&content) {
                    Ok(e) => e,
                    Err(_) => continue,
                };

                // Create node for this document element
                let node_id = element.element_id.clone();
                // Use char-based truncation to avoid slicing in middle of multibyte chars
                let label = if element.text.chars().count() > 50 {
                    format!("{}...", element.text.chars().take(50).collect::<String>())
                } else {
                    element.text.clone()
                };

                nodes.push(GraphNode {
                    id: node_id.clone(),
                    label,
                    node_type: element.element_type.clone(),
                    timestamp: element.metadata.ingested_at.clone(),
                    metadata: Some(serde_json::json!({
                        "filename": element.metadata.filename,
                        "source": element.metadata.source,
                        "text": element.text,
                    })),
                });

                // Create source node and link if source exists
                if let Some(source) = &element.metadata.source {
                    let source_id = format!("source:{}", source);

                    if !source_nodes.contains_key(&source_id) {
                        source_nodes.insert(source_id.clone(), source.clone());
                        nodes.push(GraphNode {
                            id: source_id.clone(),
                            label: source.clone(),
                            node_type: "Source".to_string(),
                            timestamp: None,
                            metadata: None,
                        });
                    }

                    links.push(GraphLink {
                        source: source_id,
                        target: node_id,
                        relationship: "contains".to_string(),
                        weight: Some(1.0),
                    });
                }
            }
        }
    }

    log::info!("Returning graph with {} nodes and {} links", nodes.len(), links.len());

    Ok(GraphData { nodes, links })
}
