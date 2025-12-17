//! Insights IPC commands.
//!
//! AGI Phase 8: Frontend Integration - Surface insights in UI
//!
//! Research Foundation:
//! - ICDA (2024): Interactive Causal Discovery
//! - CuriosityEngine: Information-gain gap detection
//! - EmergentInsights: Correlation to NL insights
//!
//! Provides access to emergent insights, knowledge gaps, and causal verification.

use serde::{Deserialize, Serialize};
use tauri::command;

// ============================================================================
// Emergent Insights
// ============================================================================

/// Insight type classification.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum InsightType {
    Correlation,
    CausalHypothesis,
    Pattern,
    Anomaly,
    Trend,
    KnowledgeGap,
}

/// Insight priority level.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum InsightPriority {
    High,
    Medium,
    Low,
}

/// An emergent insight from the intelligence engine.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmergentInsight {
    pub insight_id: String,
    pub insight_type: InsightType,
    pub title: String,
    pub description: String,
    pub confidence: f64,
    pub relevance: f64,
    pub priority: InsightPriority,
    pub source_events: Vec<String>,
    pub suggested_actions: Vec<String>,
    pub created_at: String,
    pub expires_at: Option<String>,
    pub is_read: bool,
}

/// Response from get_insights command.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InsightsResponse {
    pub success: bool,
    pub insights: Vec<EmergentInsight>,
    pub total_count: i32,
    pub unread_count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Get emergent insights.
///
/// Calls: `futurnal insights list --json [--type TYPE] [--limit N]`
#[command]
pub async fn get_insights(
    insight_type: Option<String>,
    limit: Option<i32>,
) -> Result<InsightsResponse, String> {
    let mut args = vec!["insights", "list", "--json"];

    let type_str;
    if let Some(t) = &insight_type {
        args.push("--type");
        type_str = t.clone();
        args.push(&type_str);
    }

    let limit_str;
    if let Some(l) = limit {
        args.push("--limit");
        limit_str = l.to_string();
        args.push(&limit_str);
    }

    let response: InsightsResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Get insights CLI failed: {}", e);
            return Ok(InsightsResponse {
                success: false,
                insights: vec![],
                total_count: 0,
                unread_count: 0,
                error: Some(format!("Failed to get insights: {}", e)),
            });
        }
    };

    log::info!(
        "Retrieved {} insights ({} unread)",
        response.total_count,
        response.unread_count
    );

    Ok(response)
}

/// Mark insight as read.
///
/// Calls: `futurnal insights read <insight_id> --json`
#[command]
pub async fn mark_insight_read(insight_id: String) -> Result<bool, String> {
    let args = vec!["insights", "read", &insight_id, "--json"];

    match crate::python::execute_cli::<serde_json::Value>(&args).await {
        Ok(_) => {
            log::info!("Marked insight {} as read", insight_id);
            Ok(true)
        }
        Err(e) => {
            log::warn!("Mark insight read failed: {}", e);
            Ok(false)
        }
    }
}

/// Dismiss an insight.
///
/// Calls: `futurnal insights dismiss <insight_id> --json`
#[command]
pub async fn dismiss_insight(insight_id: String) -> Result<bool, String> {
    let args = vec!["insights", "dismiss", &insight_id, "--json"];

    match crate::python::execute_cli::<serde_json::Value>(&args).await {
        Ok(_) => {
            log::info!("Dismissed insight {}", insight_id);
            Ok(true)
        }
        Err(e) => {
            log::warn!("Dismiss insight failed: {}", e);
            Ok(false)
        }
    }
}

// ============================================================================
// Knowledge Gaps
// ============================================================================

/// Knowledge gap type.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum GapType {
    IsolatedCluster,
    ForgottenMemory,
    BridgeOpportunity,
    MissingSynthesis,
    AspirationDisconnect,
}

/// A knowledge gap detected by CuriosityEngine.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KnowledgeGap {
    pub gap_id: String,
    pub gap_type: GapType,
    pub title: String,
    pub description: String,
    pub information_gain: f64,
    pub related_topics: Vec<String>,
    pub exploration_prompts: Vec<String>,
    pub created_at: String,
    pub is_addressed: bool,
}

/// Response from get_knowledge_gaps command.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KnowledgeGapsResponse {
    pub success: bool,
    pub gaps: Vec<KnowledgeGap>,
    pub total_count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Get knowledge gaps detected by CuriosityEngine.
///
/// Calls: `futurnal insights gaps --json [--limit N]`
#[command]
pub async fn get_knowledge_gaps(limit: Option<i32>) -> Result<KnowledgeGapsResponse, String> {
    let mut args = vec!["insights", "gaps", "--json"];

    let limit_str;
    if let Some(l) = limit {
        args.push("--limit");
        limit_str = l.to_string();
        args.push(&limit_str);
    }

    let response: KnowledgeGapsResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Get knowledge gaps CLI failed: {}", e);
            return Ok(KnowledgeGapsResponse {
                success: false,
                gaps: vec![],
                total_count: 0,
                error: Some(format!("Failed to get knowledge gaps: {}", e)),
            });
        }
    };

    log::info!("Retrieved {} knowledge gaps", response.total_count);

    Ok(response)
}

/// Mark knowledge gap as addressed.
///
/// Calls: `futurnal insights gap-addressed <gap_id> --json`
#[command]
pub async fn mark_gap_addressed(gap_id: String) -> Result<bool, String> {
    let args = vec!["insights", "gap-addressed", &gap_id, "--json"];

    match crate::python::execute_cli::<serde_json::Value>(&args).await {
        Ok(_) => {
            log::info!("Marked gap {} as addressed", gap_id);
            Ok(true)
        }
        Err(e) => {
            log::warn!("Mark gap addressed failed: {}", e);
            Ok(false)
        }
    }
}

// ============================================================================
// Causal Verification (ICDA)
// ============================================================================

/// Causal response type from user.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum CausalResponseType {
    YesCausal,
    NoCorrelation,
    ReverseCausation,
    Confounder,
    Uncertain,
    Skip,
}

/// A causal verification question from ICDA.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalVerificationQuestion {
    pub question_id: String,
    pub candidate_id: String,
    pub cause_event: String,
    pub effect_event: String,
    pub main_question: String,
    pub context: String,
    pub evidence_summary: String,
    pub response_options: Vec<CausalResponseOption>,
    pub initial_confidence: f64,
}

/// Response option for causal question.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalResponseOption {
    pub value: CausalResponseType,
    pub label: String,
}

/// Response from get_pending_verifications command.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PendingVerificationsResponse {
    pub success: bool,
    pub questions: Vec<CausalVerificationQuestion>,
    pub total_pending: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Get pending causal verifications.
///
/// Calls: `futurnal insights causal-pending --json [--limit N]`
#[command]
pub async fn get_pending_verifications(
    limit: Option<i32>,
) -> Result<PendingVerificationsResponse, String> {
    let mut args = vec!["insights", "causal-pending", "--json"];

    let limit_str;
    if let Some(l) = limit {
        args.push("--limit");
        limit_str = l.to_string();
        args.push(&limit_str);
    }

    let response: PendingVerificationsResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Get pending verifications CLI failed: {}", e);
            return Ok(PendingVerificationsResponse {
                success: false,
                questions: vec![],
                total_pending: 0,
                error: Some(format!("Failed to get pending verifications: {}", e)),
            });
        }
    };

    log::info!(
        "Retrieved {} pending causal verifications",
        response.total_pending
    );

    Ok(response)
}

/// Submit causal verification response.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalVerificationSubmit {
    pub question_id: String,
    pub response: CausalResponseType,
    pub explanation: Option<String>,
}

/// Response from submit_verification command.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerificationResultResponse {
    pub success: bool,
    pub candidate_id: String,
    pub new_confidence: f64,
    pub confidence_delta: f64,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Submit causal verification response.
///
/// Calls: `futurnal insights causal-verify <question_id> <response> --json [--explanation TEXT]`
#[command]
pub async fn submit_causal_verification(
    request: CausalVerificationSubmit,
) -> Result<VerificationResultResponse, String> {
    let response_str = match request.response {
        CausalResponseType::YesCausal => "yes_causal",
        CausalResponseType::NoCorrelation => "no_correlation",
        CausalResponseType::ReverseCausation => "reverse_causation",
        CausalResponseType::Confounder => "confounder",
        CausalResponseType::Uncertain => "uncertain",
        CausalResponseType::Skip => "skip",
    };

    let mut args = vec![
        "insights".to_string(),
        "causal-verify".to_string(),
        request.question_id.clone(),
        response_str.to_string(),
        "--json".to_string(),
    ];

    if let Some(explanation) = &request.explanation {
        args.push("--explanation".to_string());
        args.push(explanation.clone());
    }

    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    let response: VerificationResultResponse = match crate::python::execute_cli(&args_ref).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Submit causal verification CLI failed: {}", e);
            return Ok(VerificationResultResponse {
                success: false,
                candidate_id: String::new(),
                new_confidence: 0.0,
                confidence_delta: 0.0,
                status: "error".to_string(),
                error: Some(format!("Failed to submit verification: {}", e)),
            });
        }
    };

    log::info!(
        "Submitted causal verification for {}: {} (confidence: {:.2} -> delta: {:+.2})",
        request.question_id,
        response.status,
        response.new_confidence,
        response.confidence_delta
    );

    Ok(response)
}

// ============================================================================
// Insight Statistics
// ============================================================================

/// Insight statistics response.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InsightStatsResponse {
    pub success: bool,
    pub total_insights: i32,
    pub unread_insights: i32,
    pub total_gaps: i32,
    pub pending_verifications: i32,
    pub verified_causal_count: i32,
    pub last_scan_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Get insight statistics.
///
/// Calls: `futurnal insights stats --json`
#[command]
pub async fn get_insight_stats() -> Result<InsightStatsResponse, String> {
    let args = vec!["insights", "stats", "--json"];

    let response: InsightStatsResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Get insight stats CLI failed: {}", e);
            return Ok(InsightStatsResponse {
                success: false,
                total_insights: 0,
                unread_insights: 0,
                total_gaps: 0,
                pending_verifications: 0,
                verified_causal_count: 0,
                last_scan_at: None,
                error: Some(format!("Failed to get insight stats: {}", e)),
            });
        }
    };

    log::info!(
        "Insight stats: {} insights, {} gaps, {} pending verifications",
        response.total_insights,
        response.total_gaps,
        response.pending_verifications
    );

    Ok(response)
}

/// Trigger manual insight scan.
///
/// Calls: `futurnal insights scan --json`
#[command]
pub async fn trigger_insight_scan() -> Result<InsightStatsResponse, String> {
    let args = vec!["insights", "scan", "--json"];

    let response: InsightStatsResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Trigger insight scan CLI failed: {}", e);
            return Ok(InsightStatsResponse {
                success: false,
                total_insights: 0,
                unread_insights: 0,
                total_gaps: 0,
                pending_verifications: 0,
                verified_causal_count: 0,
                last_scan_at: None,
                error: Some(format!("Failed to trigger insight scan: {}", e)),
            });
        }
    };

    log::info!("Triggered insight scan, found {} new insights", response.unread_insights);

    Ok(response)
}
