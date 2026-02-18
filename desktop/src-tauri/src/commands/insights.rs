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
/// Must match Python InsightType enum in emergent_insights.py
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum InsightType {
    // Core types
    Correlation,
    CausalHypothesis,
    Pattern,
    Anomaly,
    Trend,
    KnowledgeGap,
    // Extended types from Python backend
    TemporalCorrelation,
    BehavioralPattern,
    AspirationMisalignment,
    ProductivityPattern,
    WeeklyRhythm,
    SequencePattern,
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

// ============================================================================
// Phase 2B: Pattern Detection
// ============================================================================

/// Day-of-week pattern data.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DayOfWeekPattern {
    pub day_index: i32,
    pub day_name: String,
    pub event_count: i32,
    pub average_count: f64,
    pub deviation_pct: f64,
    pub is_peak: bool,
    pub is_trough: bool,
    pub event_type: String,
}

/// Time-lagged correlation pattern.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimeLaggedPattern {
    pub lag_hours: i32,
    pub lag_range: String,
    pub occurrence_count: i32,
    pub avg_actual_lag_hours: f64,
    pub proportion: f64,
    pub event_type_a: String,
    pub event_type_b: String,
    pub is_significant: bool,
}

/// Response from detect_patterns command.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PatternsResponse {
    pub success: bool,
    pub time_range: Option<TimeRange>,
    pub patterns: PatternsData,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Time range for pattern detection.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimeRange {
    pub start: String,
    pub end: String,
}

/// Patterns data container.
#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct PatternsData {
    #[serde(default)]
    pub day_of_week: Vec<DayOfWeekPattern>,
    #[serde(default)]
    pub time_lagged: Vec<TimeLaggedPattern>,
}

/// Detect temporal patterns in activity data.
///
/// Phase 2B Feature: Weekly rhythm and time-lagged correlation detection.
///
/// Calls: `futurnal insights patterns --json [--type TYPE] [--days N]`
#[command]
pub async fn detect_patterns(
    pattern_type: Option<String>,
    days: Option<i32>,
    event_type: Option<String>,
) -> Result<PatternsResponse, String> {
    let mut args = vec!["insights".to_string(), "patterns".to_string(), "--json".to_string()];

    if let Some(t) = &pattern_type {
        args.push("--type".to_string());
        args.push(t.clone());
    }

    if let Some(d) = days {
        args.push("--days".to_string());
        args.push(d.to_string());
    }

    if let Some(e) = &event_type {
        args.push("--event-type".to_string());
        args.push(e.clone());
    }

    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    let response: PatternsResponse = match crate::python::execute_cli(&args_ref).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Detect patterns CLI failed: {}", e);
            return Ok(PatternsResponse {
                success: false,
                time_range: None,
                patterns: PatternsData::default(),
                error: Some(format!("Failed to detect patterns: {}", e)),
            });
        }
    };

    log::info!(
        "Detected patterns: {} day-of-week, {} time-lagged",
        response.patterns.day_of_week.len(),
        response.patterns.time_lagged.len()
    );

    Ok(response)
}

// ============================================================================
// User Insight Saving (Phase C: Save Insight)
// ============================================================================

/// Request to save a user insight from chat.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveUserInsightRequest {
    pub content: String,
    pub conversation_id: Option<String>,
    pub related_entities: Vec<String>,
    pub source: String,
}

/// Response from saving a user insight.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveUserInsightResponse {
    pub success: bool,
    pub insight_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Save a user-generated insight from chat conversation.
///
/// Calls: `futurnal insights save --content "..." --json [--conversation-id ID] [--entities e1,e2]`
#[command]
pub async fn save_user_insight(
    request: SaveUserInsightRequest,
) -> Result<SaveUserInsightResponse, String> {
    let mut args = vec![
        "insights".to_string(),
        "save".to_string(),
        "--content".to_string(),
        request.content.clone(),
        "--json".to_string(),
    ];

    if let Some(conv_id) = &request.conversation_id {
        args.push("--conversation-id".to_string());
        args.push(conv_id.clone());
    }

    if !request.related_entities.is_empty() {
        args.push("--entities".to_string());
        args.push(request.related_entities.join(","));
    }

    args.push("--source".to_string());
    args.push(request.source.clone());

    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    let response: SaveUserInsightResponse = match crate::python::execute_cli(&args_ref).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Save user insight CLI failed: {}", e);
            return Ok(SaveUserInsightResponse {
                success: false,
                insight_id: None,
                error: Some(format!("Failed to save insight: {}", e)),
            });
        }
    };

    log::info!(
        "Saved user insight: {:?} (content: {}...)",
        response.insight_id,
        &request.content[..request.content.len().min(50)]
    );

    Ok(response)
}

// ============================================================================
// Phase 2C: User Feedback Integration
// ============================================================================

/// Feedback rating type.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackRating {
    Valuable,
    NotValuable,
    Dismiss,
    Neutral,
}

/// Request to submit insight feedback.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubmitFeedbackRequest {
    pub insight_id: String,
    pub rating: FeedbackRating,
    pub insight_type: Option<String>,
    pub confidence: Option<f64>,
    pub context: Option<String>,
}

/// Response from submitting feedback.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubmitFeedbackResponse {
    pub success: bool,
    pub feedback_id: Option<String>,
    pub insight_id: String,
    pub rating: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<FeedbackStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Feedback statistics.
#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct FeedbackStats {
    #[serde(default)]
    pub valuable: i32,
    #[serde(default)]
    pub not_valuable: i32,
    #[serde(default)]
    pub dismiss: i32,
    #[serde(default)]
    pub neutral: i32,
}

/// Submit feedback on an insight.
///
/// Phase 2C Feature: User feedback for adaptive ranking.
///
/// Calls: `futurnal insights feedback <insight_id> <rating> --json [--type TYPE] [--context TEXT]`
#[command]
pub async fn submit_insight_feedback(
    request: SubmitFeedbackRequest,
) -> Result<SubmitFeedbackResponse, String> {
    let rating_str = match request.rating {
        FeedbackRating::Valuable => "valuable",
        FeedbackRating::NotValuable => "not_valuable",
        FeedbackRating::Dismiss => "dismiss",
        FeedbackRating::Neutral => "neutral",
    };

    let mut args = vec![
        "insights".to_string(),
        "feedback".to_string(),
        request.insight_id.clone(),
        rating_str.to_string(),
        "--json".to_string(),
    ];

    if let Some(t) = &request.insight_type {
        args.push("--type".to_string());
        args.push(t.clone());
    }

    if let Some(c) = request.confidence {
        args.push("--confidence".to_string());
        args.push(c.to_string());
    }

    if let Some(ctx) = &request.context {
        args.push("--context".to_string());
        args.push(ctx.clone());
    }

    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    let response: SubmitFeedbackResponse = match crate::python::execute_cli(&args_ref).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Submit feedback CLI failed: {}", e);
            return Ok(SubmitFeedbackResponse {
                success: false,
                feedback_id: None,
                insight_id: request.insight_id,
                rating: rating_str.to_string(),
                stats: None,
                error: Some(format!("Failed to submit feedback: {}", e)),
            });
        }
    };

    log::info!(
        "Submitted feedback for insight {}: {} (feedback_id: {:?})",
        request.insight_id,
        rating_str,
        response.feedback_id
    );

    Ok(response)
}

/// Response from get_feedback_stats command.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FeedbackStatsResponse {
    pub success: bool,
    pub stats: FeedbackStats,
    pub total_feedback: i32,
    pub valuable_percentage: f64,
    #[serde(default)]
    pub ranking_weights: std::collections::HashMap<String, f64>,
    #[serde(default)]
    pub type_preferences: std::collections::HashMap<String, f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Get feedback statistics and personalized ranking weights.
///
/// Phase 2C Feature: View how feedback shapes ranking.
///
/// Calls: `futurnal insights feedback-stats --json`
#[command]
pub async fn get_feedback_stats() -> Result<FeedbackStatsResponse, String> {
    let args = vec!["insights", "feedback-stats", "--json"];

    let response: FeedbackStatsResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Get feedback stats CLI failed: {}", e);
            return Ok(FeedbackStatsResponse {
                success: false,
                stats: FeedbackStats::default(),
                total_feedback: 0,
                valuable_percentage: 0.0,
                ranking_weights: std::collections::HashMap::new(),
                type_preferences: std::collections::HashMap::new(),
                error: Some(format!("Failed to get feedback stats: {}", e)),
            });
        }
    };

    log::info!(
        "Feedback stats: {} total ({:.0}% valuable)",
        response.total_feedback,
        response.valuable_percentage
    );

    Ok(response)
}
