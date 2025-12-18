//! Paper Search IPC commands.
//!
//! Phase D: Academic Paper Agent
//!
//! Research Foundation:
//! - Semantic Scholar API integration
//! - Academic paper discovery and download
//!
//! Provides paper search, metadata retrieval, and PDF download capabilities.

use serde::{Deserialize, Serialize};
use tauri::command;

// ============================================================================
// Paper Search
// ============================================================================

/// Paper author information.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PaperAuthor {
    pub name: String,
    pub author_id: Option<String>,
}

/// Paper metadata from search results.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PaperMetadata {
    pub paper_id: String,
    pub title: String,
    pub authors: Vec<PaperAuthor>,
    pub year: Option<i32>,
    pub abstract_text: Option<String>,
    pub venue: Option<String>,
    pub citation_count: Option<i32>,
    pub pdf_url: Option<String>,
    pub semantic_scholar_url: Option<String>,
    pub doi: Option<String>,
    pub arxiv_id: Option<String>,
    pub fields_of_study: Vec<String>,
}

/// Paper search result.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PaperSearchResponse {
    pub success: bool,
    pub query: String,
    pub papers: Vec<PaperMetadata>,
    pub total: i32,
    pub search_time_ms: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Search request parameters.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PaperSearchRequest {
    pub query: String,
    pub limit: Option<i32>,
    pub year_from: Option<i32>,
    pub year_to: Option<i32>,
    pub fields: Option<Vec<String>>,
}

/// Search for academic papers.
///
/// Calls: `futurnal papers search "<query>" --json [--limit N] [--year-from Y] [--year-to Y]`
#[command]
pub async fn search_papers(request: PaperSearchRequest) -> Result<PaperSearchResponse, String> {
    let mut args = vec![
        "papers".to_string(),
        "search".to_string(),
        request.query.clone(),
        "--json".to_string(),
    ];

    if let Some(limit) = request.limit {
        args.push("--limit".to_string());
        args.push(limit.to_string());
    }

    if let Some(year_from) = request.year_from {
        args.push("--year-from".to_string());
        args.push(year_from.to_string());
    }

    if let Some(year_to) = request.year_to {
        args.push("--year-to".to_string());
        args.push(year_to.to_string());
    }

    if let Some(fields) = &request.fields {
        if !fields.is_empty() {
            args.push("--fields".to_string());
            args.push(fields.join(","));
        }
    }

    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    let response: PaperSearchResponse = match crate::python::execute_cli(&args_ref).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Paper search CLI failed: {}", e);
            return Ok(PaperSearchResponse {
                success: false,
                query: request.query,
                papers: vec![],
                total: 0,
                search_time_ms: None,
                error: Some(format!("Failed to search papers: {}", e)),
            });
        }
    };

    log::info!(
        "Paper search '{}' returned {} results",
        request.query,
        response.papers.len()
    );

    Ok(response)
}

// ============================================================================
// Paper Download
// ============================================================================

/// Downloaded paper information.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadedPaper {
    pub paper_id: String,
    pub title: String,
    pub local_path: String,
    pub file_size_bytes: i64,
}

/// Paper download response.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PaperDownloadResponse {
    pub success: bool,
    pub downloaded: Option<DownloadedPaper>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Download request for a paper.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PaperDownloadRequest {
    pub paper_id: String,
    pub pdf_url: String,
    pub title: String,
    pub year: Option<i32>,
}

/// Download a paper PDF.
///
/// Calls: `futurnal papers download "<paper_id>" --pdf-url "<url>" --title "<title>" --json`
#[command]
pub async fn download_paper(request: PaperDownloadRequest) -> Result<PaperDownloadResponse, String> {
    let mut args = vec![
        "papers".to_string(),
        "download".to_string(),
        request.paper_id.clone(),
        "--pdf-url".to_string(),
        request.pdf_url.clone(),
        "--title".to_string(),
        request.title.clone(),
        "--json".to_string(),
    ];

    if let Some(year) = request.year {
        args.push("--year".to_string());
        args.push(year.to_string());
    }

    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    let response: PaperDownloadResponse = match crate::python::execute_cli(&args_ref).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Paper download CLI failed: {}", e);
            return Ok(PaperDownloadResponse {
                success: false,
                downloaded: None,
                error: Some(format!("Failed to download paper: {}", e)),
            });
        }
    };

    if let Some(ref paper) = response.downloaded {
        log::info!("Downloaded paper: {} to {}", paper.title, paper.local_path);
    }

    Ok(response)
}

// ============================================================================
// Paper Recommendations
// ============================================================================

/// Paper recommendations response.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PaperRecommendationsResponse {
    pub success: bool,
    pub source_paper_id: String,
    pub recommendations: Vec<PaperMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Get paper recommendations based on a paper.
///
/// Calls: `futurnal papers recommend "<paper_id>" --json [--limit N]`
#[command]
pub async fn get_paper_recommendations(
    paper_id: String,
    limit: Option<i32>,
) -> Result<PaperRecommendationsResponse, String> {
    let mut args = vec![
        "papers".to_string(),
        "recommend".to_string(),
        paper_id.clone(),
        "--json".to_string(),
    ];

    if let Some(l) = limit {
        args.push("--limit".to_string());
        args.push(l.to_string());
    }

    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    let response: PaperRecommendationsResponse = match crate::python::execute_cli(&args_ref).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Paper recommendations CLI failed: {}", e);
            return Ok(PaperRecommendationsResponse {
                success: false,
                source_paper_id: paper_id,
                recommendations: vec![],
                error: Some(format!("Failed to get recommendations: {}", e)),
            });
        }
    };

    log::info!(
        "Got {} recommendations for paper {}",
        response.recommendations.len(),
        paper_id
    );

    Ok(response)
}

/// Get details for a specific paper.
///
/// Calls: `futurnal papers get "<paper_id>" --json`
#[command]
pub async fn get_paper_details(paper_id: String) -> Result<PaperMetadata, String> {
    let args = vec!["papers", "get", &paper_id, "--json"];

    match crate::python::execute_cli::<PaperMetadata>(&args).await {
        Ok(paper) => {
            log::info!("Retrieved paper details: {}", paper.title);
            Ok(paper)
        }
        Err(e) => {
            log::warn!("Get paper details CLI failed: {}", e);
            Err(format!("Failed to get paper details: {}", e))
        }
    }
}

// ============================================================================
// Agentic Paper Search
// ============================================================================

/// Search strategy information.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SearchStrategy {
    pub query: String,
    #[serde(rename = "type")]
    pub strategy_type: String,
    pub rationale: String,
}

/// Scored paper with relevance information.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ScoredPaper {
    pub paper_id: String,
    pub title: String,
    pub authors: Vec<PaperAuthor>,
    pub year: Option<i32>,
    pub abstract_text: Option<String>,
    pub venue: Option<String>,
    pub citation_count: Option<i32>,
    pub pdf_url: Option<String>,
    pub source_url: Option<String>,
    pub relevance_score: f64,
    pub rationale: String,
}

/// Agentic search response with synthesis and suggestions.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgenticSearchResponse {
    pub success: bool,
    pub query: String,
    pub papers: Vec<ScoredPaper>,
    pub total_evaluated: i32,
    pub synthesis: String,
    pub suggestions: Vec<String>,
    pub strategies_tried: Vec<SearchStrategy>,
    pub search_time_ms: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Agentic search request.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgenticSearchRequest {
    pub query: String,
}

/// Intelligent agentic paper search.
///
/// Calls: `futurnal papers agentic-search "<query>" --json`
///
/// Features:
/// - Query analysis and understanding
/// - Multiple search strategies (synonyms, expansions)
/// - Relevance scoring
/// - Synthesis and suggestions
#[command]
pub async fn agentic_search_papers(request: AgenticSearchRequest) -> Result<AgenticSearchResponse, String> {
    log::info!("Starting agentic paper search: {}", request.query);

    let args = vec![
        "papers".to_string(),
        "agentic-search".to_string(),
        request.query.clone(),
        "--json".to_string(),
    ];

    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    // Use longer timeout for agentic search (involves multiple API calls and rate limiting)
    let response: AgenticSearchResponse = match crate::python::execute_cli_with_timeout(&args_ref, 120).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Agentic paper search CLI failed: {}", e);
            return Ok(AgenticSearchResponse {
                success: false,
                query: request.query,
                papers: vec![],
                total_evaluated: 0,
                synthesis: String::new(),
                suggestions: vec![],
                strategies_tried: vec![],
                search_time_ms: None,
                error: Some(format!("Failed to search papers: {}", e)),
            });
        }
    };

    log::info!(
        "Agentic search '{}' found {} relevant papers (evaluated {})",
        request.query,
        response.papers.len(),
        response.total_evaluated
    );

    Ok(response)
}

// ============================================================================
// Paper Ingestion
// ============================================================================

/// Paper ingestion request.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PaperIngestRequest {
    pub paper_ids: Vec<String>,
}

/// Ingested paper info.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IngestedPaperInfo {
    pub paper_id: String,
    pub title: String,
    pub status: String,
}

/// Paper ingestion response.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PaperIngestResponse {
    pub success: bool,
    pub queued: i32,
    pub papers: Vec<IngestedPaperInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Ingest papers into the knowledge graph.
///
/// Calls: `futurnal papers ingest <paper_ids...> --json`
#[command]
pub async fn ingest_papers(request: PaperIngestRequest) -> Result<PaperIngestResponse, String> {
    log::info!("Ingesting {} papers into knowledge graph", request.paper_ids.len());

    let mut args = vec![
        "papers".to_string(),
        "ingest".to_string(),
    ];

    // Add paper IDs
    args.extend(request.paper_ids.iter().cloned());
    args.push("--json".to_string());

    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    let response: PaperIngestResponse = match crate::python::execute_cli(&args_ref).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Paper ingest CLI failed: {}", e);
            return Ok(PaperIngestResponse {
                success: false,
                queued: 0,
                papers: vec![],
                error: Some(format!("Failed to ingest papers: {}", e)),
            });
        }
    };

    log::info!("Queued {} papers for ingestion", response.queued);

    Ok(response)
}

// ============================================================================
// Paper Status
// ============================================================================

/// Paper status information.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PaperStatusInfo {
    pub paper_id: String,
    pub title: String,
    pub local_path: Option<String>,
    pub download_status: String,
    pub ingestion_status: String,
    pub downloaded_at: Option<String>,
    pub ingested_at: Option<String>,
    pub file_size_bytes: i64,
}

/// Paper status response.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PaperStatusResponse {
    pub success: bool,
    pub paper: Option<PaperStatusInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// All papers status response.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AllPapersStatusResponse {
    pub success: bool,
    pub total: i32,
    pub counts: PaperStatusCounts,
    pub papers: Vec<PaperStatusInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Status counts by category.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PaperStatusCounts {
    pub download: std::collections::HashMap<String, i32>,
    pub ingestion: std::collections::HashMap<String, i32>,
}

/// Get status for a specific paper.
///
/// Calls: `futurnal papers status <paper_id> --json`
#[command]
pub async fn get_paper_status(paper_id: String) -> Result<PaperStatusResponse, String> {
    let args = vec!["papers", "status", &paper_id, "--json"];

    let response: PaperStatusResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Paper status CLI failed: {}", e);
            return Ok(PaperStatusResponse {
                success: false,
                paper: None,
                error: Some(format!("Failed to get paper status: {}", e)),
            });
        }
    };

    if let Some(ref paper) = response.paper {
        log::info!(
            "Paper status: {} - download: {}, ingestion: {}",
            paper.paper_id,
            paper.download_status,
            paper.ingestion_status
        );
    }

    Ok(response)
}

/// Get status for all papers.
///
/// Calls: `futurnal papers status --all --json`
#[command]
pub async fn get_all_papers_status() -> Result<AllPapersStatusResponse, String> {
    let args = vec!["papers", "status", "--all", "--json"];

    let response: AllPapersStatusResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("All papers status CLI failed: {}", e);
            return Ok(AllPapersStatusResponse {
                success: false,
                total: 0,
                counts: PaperStatusCounts {
                    download: std::collections::HashMap::new(),
                    ingestion: std::collections::HashMap::new(),
                },
                papers: vec![],
                error: Some(format!("Failed to get papers status: {}", e)),
            });
        }
    };

    log::info!("Retrieved status for {} papers", response.total);

    Ok(response)
}
