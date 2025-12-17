//! Schema Stats IPC commands.
//!
//! Step 08: Frontend Intelligence Integration - Phase 5
//!
//! Research Foundation:
//! - GFM-RAG: Schema-aware graph construction
//! - ACE: Adaptive schema evolution
//!
//! Provides schema evolution statistics from PKG.

use serde::{Deserialize, Serialize};
use tauri::command;

/// Entity type statistics.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntityTypeStat {
    #[serde(rename = "type")]
    pub entity_type: String,
    pub count: i32,
    pub first_seen: Option<String>,
    pub last_seen: Option<String>,
}

/// Relationship type statistics.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationshipTypeStat {
    #[serde(rename = "type")]
    pub rel_type: String,
    pub count: i32,
    pub confidence_avg: f64,
}

/// Quality metrics.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QualityMetrics {
    pub precision: f64,
    pub recall: f64,
    pub temporal_accuracy: f64,
}

/// Schema evolution event.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SchemaEvolutionEvent {
    pub timestamp: Option<String>,
    pub change_type: String,
    pub details: String,
}

/// Response from schema stats command.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SchemaStatsResponse {
    pub success: bool,
    pub entity_types: Vec<EntityTypeStat>,
    pub relationship_types: Vec<RelationshipTypeStat>,
    pub quality_metrics: QualityMetrics,
    pub evolution_timeline: Vec<SchemaEvolutionEvent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Get schema statistics.
///
/// Step 08: Schema Evolution Dashboard
///
/// Calls: `futurnal schema stats --json`
#[command]
pub async fn get_schema_stats() -> Result<SchemaStatsResponse, String> {
    let args = vec!["schema", "stats", "--json"];

    let response: SchemaStatsResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Schema stats CLI failed: {}", e);
            return Ok(SchemaStatsResponse {
                success: false,
                entity_types: vec![],
                relationship_types: vec![],
                quality_metrics: QualityMetrics {
                    precision: 0.0,
                    recall: 0.0,
                    temporal_accuracy: 0.0,
                },
                evolution_timeline: vec![],
                error: Some(format!("Schema stats failed: {}", e)),
            });
        }
    };

    log::info!(
        "Retrieved schema stats: {} entity types, {} relationship types",
        response.entity_types.len(),
        response.relationship_types.len()
    );

    Ok(response)
}
