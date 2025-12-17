//! Learning Progress IPC commands.
//!
//! Step 08: Frontend Intelligence Integration - Phase 6
//!
//! Research Foundation:
//! - RLHI: Reinforcement Learning from Human Interactions
//! - AgentFlow: Learning from user feedback
//! - Option B: Ghost frozen, learning via token priors
//!
//! Provides experiential learning progress metrics and document recording.

use serde::{Deserialize, Serialize};
use tauri::command;
use uuid::Uuid;

/// Quality progression metrics.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QualityProgression {
    pub before: f64,
    pub after: f64,
    pub improvement: f64,
}

/// Pattern learning statistics.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PatternLearning {
    pub entity_priors: i32,
    pub relation_priors: i32,
    pub temporal_priors: i32,
}

/// Quality gates status.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QualityGates {
    pub ghost_frozen: bool,
    pub improvement_threshold: f64,
    pub meets_threshold: bool,
}

/// Response from learning progress command.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LearningProgressResponse {
    pub success: bool,
    pub documents_processed: i32,
    pub success_rate: f64,
    pub quality_progression: QualityProgression,
    pub pattern_learning: PatternLearning,
    pub quality_gates: QualityGates,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Get learning progress metrics.
///
/// Step 08: Learning Progress Indicator
///
/// Calls: `futurnal learning progress --json`
#[command]
pub async fn get_learning_progress() -> Result<LearningProgressResponse, String> {
    let args = vec!["learning", "progress", "--json"];

    let response: LearningProgressResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Learning progress CLI failed: {}", e);
            return Ok(LearningProgressResponse {
                success: false,
                documents_processed: 0,
                success_rate: 0.0,
                quality_progression: QualityProgression {
                    before: 0.0,
                    after: 0.0,
                    improvement: 0.0,
                },
                pattern_learning: PatternLearning {
                    entity_priors: 0,
                    relation_priors: 0,
                    temporal_priors: 0,
                },
                quality_gates: QualityGates {
                    ghost_frozen: true,
                    improvement_threshold: 0.05,
                    meets_threshold: false,
                },
                error: Some(format!("Learning progress failed: {}", e)),
            });
        }
    };

    log::info!(
        "Retrieved learning progress: {} documents, {:.1}% success rate",
        response.documents_processed,
        response.success_rate * 100.0
    );

    Ok(response)
}

/// Request payload for recording document learning.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordDocumentRequest {
    /// Content extracted from the document.
    pub content: String,
    /// Source of the document (chat, ingestion, etc.).
    #[serde(default = "default_source")]
    pub source: String,
    /// Content type (text, image, audio, document).
    #[serde(default = "default_content_type")]
    pub content_type: String,
    /// Whether extraction was successful.
    #[serde(default = "default_success")]
    pub success: bool,
    /// Optional quality score (0-1).
    pub quality_score: Option<f64>,
    /// Optional entity types discovered.
    pub entity_types: Option<Vec<String>>,
    /// Optional relation types discovered.
    pub relation_types: Option<Vec<String>>,
}

fn default_source() -> String {
    "chat".to_string()
}

fn default_content_type() -> String {
    "text".to_string()
}

fn default_success() -> bool {
    true
}

/// Response from record document learning command.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordDocumentResponse {
    pub success: bool,
    pub document_id: String,
    pub quality_score: f64,
    pub total_documents: i32,
    pub overall_success_rate: f64,
    pub entity_priors: i32,
    pub relation_priors: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Record a document processing event to the learning pipeline.
///
/// Step 08: Full Learning Integration
///
/// Updates learning state and token priors, then persists to disk.
/// Called by frontend when processing chat attachments.
///
/// Calls: `futurnal learning record <document_id> --json [options]`
#[command]
pub async fn record_document_learning(
    request: RecordDocumentRequest,
) -> Result<RecordDocumentResponse, String> {
    // Generate unique document ID
    let document_id = format!("doc_{}", Uuid::new_v4().to_string().replace("-", "")[..12].to_string());

    let mut args = vec![
        "learning".to_string(),
        "record".to_string(),
        document_id.clone(),
        "--json".to_string(),
        "--source".to_string(),
        request.source.clone(),
        "--type".to_string(),
        request.content_type.clone(),
    ];

    // Add content if provided
    if !request.content.is_empty() {
        args.push("--content".to_string());
        args.push(request.content.clone());
    }

    // Add success/failure flag
    if !request.success {
        args.push("--failure".to_string());
    }

    // Add quality score if provided
    if let Some(quality) = request.quality_score {
        args.push("--quality".to_string());
        args.push(format!("{:.2}", quality));
    }

    // Add entity types if provided
    if let Some(entities) = &request.entity_types {
        if !entities.is_empty() {
            args.push("--entities".to_string());
            args.push(entities.join(","));
        }
    }

    // Add relation types if provided
    if let Some(relations) = &request.relation_types {
        if !relations.is_empty() {
            args.push("--relations".to_string());
            args.push(relations.join(","));
        }
    }

    // Convert to &str for CLI execution
    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    let response: RecordDocumentResponse = match crate::python::execute_cli(&args_ref).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Record document learning CLI failed: {}", e);
            return Ok(RecordDocumentResponse {
                success: false,
                document_id,
                quality_score: 0.0,
                total_documents: 0,
                overall_success_rate: 0.0,
                entity_priors: 0,
                relation_priors: 0,
                error: Some(format!("Record document learning failed: {}", e)),
            });
        }
    };

    log::info!(
        "Recorded document {}: quality={:.2}, total={}",
        response.document_id,
        response.quality_score,
        response.total_documents
    );

    Ok(response)
}
