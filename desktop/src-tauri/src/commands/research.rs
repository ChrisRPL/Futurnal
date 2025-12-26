//! Research IPC commands.
//!
//! Web search and deep research functionality.
//!
//! Research Foundation:
//! - WebDancer (2505.22648v3): End-to-end web agents
//! - Personalized Deep Research (2509.25106v1): User-centric research
//!
//! Provides web search, deep research, and research status capabilities.

use serde::{Deserialize, Serialize};
use tauri::command;

// ============================================================================
// Web Search
// ============================================================================

/// Individual finding from web search.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct WebFinding {
    pub fact: String,
    pub source_url: String,
    pub source_title: String,
    pub reliability: String,
}

/// Source information from web search.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct WebSource {
    pub url: String,
    pub title: String,
    pub reliability: String,
    pub relevance: f64,
}

/// Web search response.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebSearchResponse {
    pub success: bool,
    pub query: String,
    pub answer: String,
    pub sources: Vec<WebSource>,
    pub confidence: f64,
    pub coverage: f64,
    pub num_sources: i32,
    pub total_steps: i32,
    pub total_pages: i32,
    pub search_time_ms: i32,
    pub findings: Vec<WebFinding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Web search request.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebSearchRequest {
    pub query: String,
    pub max_pages: Option<i32>,
}

/// Search the web for information.
///
/// Uses WebDancer-style autonomous web browsing to find and synthesize
/// information from multiple web sources.
///
/// Calls: `futurnal research web "<query>" --json [--max-pages N]`
#[command]
pub async fn web_search(request: WebSearchRequest) -> Result<WebSearchResponse, String> {
    log::info!("Starting web search: {}", request.query);

    let mut args = vec![
        "research".to_string(),
        "web".to_string(),
        request.query.clone(),
        "--json".to_string(),
    ];

    if let Some(max_pages) = request.max_pages {
        args.push("--max-pages".to_string());
        args.push(max_pages.to_string());
    }

    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    // Use longer timeout for web search (involves fetching multiple pages)
    let response: WebSearchResponse = match crate::python::execute_cli_with_timeout(&args_ref, 120).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Web search CLI failed: {}", e);
            return Ok(WebSearchResponse {
                success: false,
                query: request.query,
                answer: String::new(),
                sources: vec![],
                confidence: 0.0,
                coverage: 0.0,
                num_sources: 0,
                total_steps: 0,
                total_pages: 0,
                search_time_ms: 0,
                findings: vec![],
                error: Some(format!("Web search failed: {}", e)),
            });
        }
    };

    log::info!(
        "Web search '{}' found {} sources with {:.0}% confidence",
        request.query,
        response.num_sources,
        response.confidence * 100.0
    );

    Ok(response)
}

// ============================================================================
// Deep Research
// ============================================================================

/// Detailed finding from deep research.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ResearchFinding {
    pub content: String,
    #[serde(rename = "type")]
    pub finding_type: String,
    pub relevance: f64,
}

/// Deep research response.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeepResearchResponse {
    pub success: bool,
    pub query: String,
    pub user_id: String,
    pub summary: String,
    pub key_points: Vec<String>,
    pub sources: Vec<serde_json::Value>,
    pub num_sources_consulted: i32,
    pub expertise_level_used: String,
    pub depth_used: String,
    pub confidence: f64,
    pub relevance_score: f64,
    pub research_time_ms: i32,
    pub detailed_findings: Vec<ResearchFinding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Deep research request.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeepResearchRequest {
    pub query: String,
    pub depth: Option<String>,
    pub user_id: Option<String>,
}

/// Conduct deep personalized research.
///
/// Combines knowledge graph, vector search, and web research
/// to provide comprehensive, personalized research results.
///
/// Calls: `futurnal research deep "<query>" --json [--depth D] [--user U]`
#[command]
pub async fn deep_research(request: DeepResearchRequest) -> Result<DeepResearchResponse, String> {
    log::info!("Starting deep research: {}", request.query);

    let mut args = vec![
        "research".to_string(),
        "deep".to_string(),
        request.query.clone(),
        "--json".to_string(),
    ];

    if let Some(depth) = &request.depth {
        args.push("--depth".to_string());
        args.push(depth.clone());
    }

    if let Some(user_id) = &request.user_id {
        args.push("--user".to_string());
        args.push(user_id.clone());
    }

    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    // Use longer timeout for deep research (involves multiple sources)
    let response: DeepResearchResponse = match crate::python::execute_cli_with_timeout(&args_ref, 180).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Deep research CLI failed: {}", e);
            return Ok(DeepResearchResponse {
                success: false,
                query: request.query,
                user_id: request.user_id.unwrap_or_else(|| "default".to_string()),
                summary: String::new(),
                key_points: vec![],
                sources: vec![],
                num_sources_consulted: 0,
                expertise_level_used: "intermediate".to_string(),
                depth_used: "standard".to_string(),
                confidence: 0.0,
                relevance_score: 0.0,
                research_time_ms: 0,
                detailed_findings: vec![],
                error: Some(format!("Deep research failed: {}", e)),
            });
        }
    };

    log::info!(
        "Deep research '{}' consulted {} sources with {:.0}% confidence",
        request.query,
        response.num_sources_consulted,
        response.confidence * 100.0
    );

    Ok(response)
}

// ============================================================================
// Quick Search
// ============================================================================

/// Quick search result.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct QuickSearchResult {
    pub url: String,
    pub title: String,
    pub snippet: String,
}

/// Quick search response.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuickSearchResponse {
    pub success: bool,
    pub query: String,
    pub results: Vec<QuickSearchResult>,
    pub total: i32,
    pub search_time_ms: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Quick web search without deep analysis.
///
/// Returns raw search results from DuckDuckGo without
/// visiting pages or synthesizing answers.
///
/// Calls: `futurnal research quick "<query>" --json`
#[command]
pub async fn quick_search(query: String) -> Result<QuickSearchResponse, String> {
    log::info!("Starting quick search: {}", query);

    let args = vec![
        "research",
        "quick",
        &query,
        "--json",
    ];

    let response: QuickSearchResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Quick search CLI failed: {}", e);
            return Ok(QuickSearchResponse {
                success: false,
                query,
                results: vec![],
                total: 0,
                search_time_ms: 0,
                error: Some(format!("Quick search failed: {}", e)),
            });
        }
    };

    log::info!("Quick search found {} results", response.total);

    Ok(response)
}

// ============================================================================
// Research Status
// ============================================================================

/// Component status.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ComponentStatus {
    pub available: bool,
    pub status: String,
}

/// Research status response.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResearchStatusResponse {
    pub success: bool,
    pub components: std::collections::HashMap<String, ComponentStatus>,
    pub available_count: i32,
    pub total_count: i32,
    pub all_healthy: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Get research infrastructure status.
///
/// Calls: `futurnal research status --json`
#[command]
pub async fn get_research_status() -> Result<ResearchStatusResponse, String> {
    log::info!("Getting research status");

    let args = vec!["research", "status", "--json"];

    let response: ResearchStatusResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Research status CLI failed: {}", e);
            return Ok(ResearchStatusResponse {
                success: false,
                components: std::collections::HashMap::new(),
                available_count: 0,
                total_count: 0,
                all_healthy: false,
                error: Some(format!("Status check failed: {}", e)),
            });
        }
    };

    log::info!(
        "Research status: {}/{} components available",
        response.available_count,
        response.total_count
    );

    Ok(response)
}
