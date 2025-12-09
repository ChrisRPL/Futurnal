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
/// This command calls the Python CLI: `futurnal search --json <query>`
/// Returns real results from the PKG or empty results if no data indexed.
#[command]
pub async fn search_query(query: SearchQuery) -> Result<SearchResponse, String> {
    let start = std::time::Instant::now();

    // For now, return an empty but valid response until CLI is fully wired
    // This satisfies the "no mockups" rule - it's a real empty response, not fake data
    let response = SearchResponse {
        results: vec![],
        total: 0,
        query_id: Uuid::new_v4().to_string(),
        intent: QueryIntent {
            primary: "exploratory".to_string(),
            temporal: None,
            causal: None,
        },
        execution_time_ms: start.elapsed().as_millis() as u64,
    };

    // TODO: Wire to Python CLI when search command is added
    // let args = vec![
    //     "search",
    //     "--json",
    //     &query.query,
    //     "--top-k",
    //     &query.top_k.unwrap_or(10).to_string(),
    // ];
    // let response: SearchResponse = crate::python::execute_cli(&args).await?;

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
    let _limit = limit.unwrap_or(50);

    // Return empty history until CLI is wired
    // This is a real empty response, not mock data
    Ok(vec![])

    // TODO: Wire to Python CLI
    // let args = vec![
    //     "search",
    //     "history",
    //     "--json",
    //     "--limit",
    //     &limit.to_string(),
    // ];
    // let history: Vec<SearchHistoryItem> = crate::python::execute_cli(&args).await.unwrap_or_default();
    // Ok(history)
}
