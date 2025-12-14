//! Search IPC commands.
//!
//! Handles search queries against the Futurnal Hybrid Search API.

use serde::{Deserialize, Serialize};
use tauri::command;
use uuid::Uuid;

/// Search query input from frontend.
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchQuery {
    pub query: String,
    pub top_k: Option<u32>,
    pub filters: Option<SearchFilters>,
}

/// Optional search filters.
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchFilters {
    pub entity_types: Option<Vec<String>>,
    pub source_types: Option<Vec<String>>,
    pub date_range: Option<DateRange>,
    pub sources: Option<Vec<String>>,
}

/// Date range filter.
#[derive(Debug, Serialize, Deserialize)]
pub struct DateRange {
    pub start: Option<String>,
    pub end: Option<String>,
}

/// Individual search result.
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub content: String,
    pub score: f64,
    pub confidence: f64,
    pub timestamp: Option<String>,
    pub entity_type: Option<String>,
    pub source_type: Option<String>,
    pub source_confidence: Option<f64>,
    pub causal_chain: Option<CausalChain>,
    pub metadata: serde_json::Value,
    /// Graph context from GraphRAG traversal (per GFM-RAG paper)
    pub graph_context: Option<GraphContext>,
    /// Vector similarity score from semantic search (0-1)
    pub vector_score: Option<f64>,
    /// Graph traversal score from graph expansion (0-1)
    pub graph_score: Option<f64>,
}

/// Graph traversal context for GraphRAG results.
///
/// Per GFM-RAG paper (2502.01113v1):
/// - Shows "why" a result is relevant via graph connections
/// - Enables path visualization for user understanding
#[derive(Debug, Serialize, Deserialize)]
pub struct GraphContext {
    /// Entities connected via graph traversal
    #[serde(rename = "related_entities")]
    pub related_entities: Vec<RelatedEntity>,
    /// Relationships traversed to reach this result
    pub relationships: Vec<Relationship>,
    /// Path from query entity to this result
    #[serde(rename = "path_to_query")]
    pub path_to_query: Vec<String>,
    /// Number of hops from seed entities
    #[serde(rename = "hop_count")]
    pub hop_count: u32,
    /// Confidence score for the traversal path
    #[serde(rename = "path_confidence")]
    pub path_confidence: f64,
}

/// Related entity in graph context.
#[derive(Debug, Serialize, Deserialize)]
pub struct RelatedEntity {
    pub id: String,
    #[serde(rename = "type")]
    pub entity_type: String,
    pub name: Option<String>,
}

/// Relationship in graph context.
#[derive(Debug, Serialize, Deserialize)]
pub struct Relationship {
    #[serde(rename = "type")]
    pub rel_type: String,
    #[serde(rename = "from_entity")]
    pub from_entity: Option<String>,
    #[serde(rename = "to_entity")]
    pub to_entity: Option<String>,
    pub confidence: Option<f64>,
}

/// Causal chain information for causal search results.
#[derive(Debug, Serialize, Deserialize)]
pub struct CausalChain {
    pub anchor: String,
    pub causes: Vec<String>,
    pub effects: Vec<String>,
}

/// Search response returned to frontend.
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub total: u32,
    pub query_id: String,
    pub intent: QueryIntent,
    pub execution_time_ms: u64,
}

/// Detected query intent.
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryIntent {
    pub primary: String,
    pub temporal: Option<TemporalIntent>,
    pub causal: Option<bool>,
}

/// Temporal intent details.
#[derive(Debug, Serialize, Deserialize)]
pub struct TemporalIntent {
    pub range_type: String,
    pub start: Option<String>,
    pub end: Option<String>,
}

/// Search history item.
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchHistoryItem {
    pub id: String,
    pub query: String,
    pub timestamp: String,
    pub result_count: u32,
}

/// Execute a search query against the Hybrid Search API.
///
/// This command calls the Python CLI: `futurnal search query --json <query>`
/// Returns real results from the PKG or empty results if no data indexed.
#[command]
pub async fn search_query(query: SearchQuery) -> Result<SearchResponse, String> {
    let start = std::time::Instant::now();

    // Build CLI args
    let top_k_str = query.top_k.unwrap_or(10).to_string();
    let args = vec![
        "search",
        "query",
        &query.query,
        "--top-k",
        &top_k_str,
        "--json",
    ];

    // Execute Python CLI
    let response: SearchResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Search CLI failed, returning empty results: {}", e);
            // Return empty results on error (graceful degradation)
            SearchResponse {
                results: vec![],
                total: 0,
                query_id: Uuid::new_v4().to_string(),
                intent: QueryIntent {
                    primary: "exploratory".to_string(),
                    temporal: None,
                    causal: None,
                },
                execution_time_ms: start.elapsed().as_millis() as u64,
            }
        }
    };

    log::info!(
        "Search query '{}' returned {} results in {}ms",
        query.query,
        response.total,
        response.execution_time_ms
    );

    Ok(response)
}

/// Get search history.
///
/// Returns recent search queries and their result counts.
#[command]
pub async fn get_search_history(limit: Option<u32>) -> Result<Vec<SearchHistoryItem>, String> {
    let limit_str = limit.unwrap_or(50).to_string();

    let args = vec![
        "search",
        "history",
        "--limit",
        &limit_str,
        "--json",
    ];

    let history: Vec<SearchHistoryItem> = match crate::python::execute_cli(&args).await {
        Ok(h) => h,
        Err(e) => {
            log::warn!("Search history failed: {}", e);
            vec![]
        }
    };

    Ok(history)
}
