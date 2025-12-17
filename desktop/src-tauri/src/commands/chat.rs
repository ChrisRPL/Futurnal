//! Chat IPC commands.
//!
//! Step 03: Chat Interface & Conversational AI
//!
//! Research Foundation:
//! - ProPerSim (2509.21730v1): Proactive + personalized AI
//! - Causal-Copilot (2504.13263v2): Natural language causal exploration
//!
//! Handles conversational interface to the Personal Knowledge Graph.

use serde::{Deserialize, Serialize};
use tauri::command;
use uuid::Uuid;

/// Chat message from frontend.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatRequest {
    pub session_id: String,
    pub message: String,
    pub context_entity_id: Option<String>,
    pub model: Option<String>,
}

/// Individual chat message.
///
/// Research Foundation:
/// - ProPerSim: Session tracking with timestamps
/// - Causal-Copilot: Confidence scoring per response
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub sources: Vec<String>,
    #[serde(rename = "entityRefs")]
    pub entity_refs: Vec<String>,
    pub confidence: f64,
    pub timestamp: String,
}

/// Response from chat send command.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    pub success: bool,
    #[serde(rename = "sessionId")]
    pub session_id: String,
    pub response: ChatMessage,
}

/// Chat session info.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatSessionInfo {
    pub id: String,
    #[serde(rename = "messageCount")]
    pub message_count: u32,
    #[serde(rename = "createdAt")]
    pub created_at: String,
    #[serde(rename = "updatedAt")]
    pub updated_at: String,
    pub preview: String,
}

/// Chat history response.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatHistoryResponse {
    pub success: bool,
    #[serde(rename = "sessionId")]
    pub session_id: String,
    pub messages: Vec<ChatMessage>,
    pub total: u32,
}

/// Sessions list response.
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionsListResponse {
    pub success: bool,
    pub sessions: Vec<ChatSessionInfo>,
    pub total: u32,
}

/// Generic operation response.
#[derive(Debug, Serialize, Deserialize)]
pub struct OperationResponse {
    pub success: bool,
    #[serde(rename = "sessionId")]
    pub session_id: String,
    pub message: String,
}

/// Send a chat message and get a response.
///
/// Step 03: Chat Interface & Conversational AI
///
/// Research Foundation:
/// - ProPerSim: Multi-turn context carryover
/// - Causal-Copilot: Confidence scoring, source citations
///
/// Calls: `futurnal chat send <session_id> <message> --json`
#[command]
pub async fn send_chat_message(request: ChatRequest) -> Result<ChatResponse, String> {
    let mut args = vec![
        "chat".to_string(),
        "send".to_string(),
        request.session_id.clone(),
        request.message.clone(),
        "--json".to_string(),
    ];

    // Add context entity if provided
    if let Some(entity_id) = &request.context_entity_id {
        args.push("--context".to_string());
        args.push(entity_id.clone());
    }

    // Add model if specified
    if let Some(m) = &request.model {
        args.push("--model".to_string());
        args.push(m.clone());
    }

    // Convert to &str slice for execute_cli
    let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    // Execute Python CLI with longer timeout for LLM inference
    let response: ChatResponse = match crate::python::execute_cli_with_timeout(&args_refs, 120).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Chat send CLI failed: {}", e);
            return Err(format!("Chat send failed: {}", e));
        }
    };

    log::info!(
        "Chat message sent to session '{}', confidence: {}",
        request.session_id,
        response.response.confidence
    );

    Ok(response)
}

/// Get conversation history for a session.
///
/// Calls: `futurnal chat history <session_id> --json`
#[command]
pub async fn get_chat_history(
    session_id: String,
    limit: Option<u32>,
) -> Result<ChatHistoryResponse, String> {
    let limit_str = limit.unwrap_or(50).to_string();

    let args = vec![
        "chat",
        "history",
        &session_id,
        "--limit",
        &limit_str,
        "--json",
    ];

    let response: ChatHistoryResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Chat history CLI failed: {}", e);
            // Return empty history on error
            ChatHistoryResponse {
                success: false,
                session_id: session_id.clone(),
                messages: vec![],
                total: 0,
            }
        }
    };

    log::info!(
        "Chat history for session '{}': {} messages",
        session_id,
        response.total
    );

    Ok(response)
}

/// List all chat sessions.
///
/// Calls: `futurnal chat sessions --json`
#[command]
pub async fn list_chat_sessions(limit: Option<u32>) -> Result<SessionsListResponse, String> {
    let limit_str = limit.unwrap_or(20).to_string();

    let args = vec!["chat", "sessions", "--limit", &limit_str, "--json"];

    let response: SessionsListResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Chat sessions CLI failed: {}", e);
            // Return empty list on error
            SessionsListResponse {
                success: false,
                sessions: vec![],
                total: 0,
            }
        }
    };

    log::info!("Listed {} chat sessions", response.total);

    Ok(response)
}

/// Clear messages from a session.
///
/// Calls: `futurnal chat clear <session_id> --json`
#[command]
pub async fn clear_chat_session(session_id: String) -> Result<OperationResponse, String> {
    let args = vec!["chat", "clear", &session_id, "--json"];

    let response: OperationResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Chat clear CLI failed: {}", e);
            return Err(format!("Chat clear failed: {}", e));
        }
    };

    log::info!("Cleared chat session '{}'", session_id);

    Ok(response)
}

/// Delete a chat session.
///
/// Calls: `futurnal chat delete <session_id> --json`
#[command]
pub async fn delete_chat_session(session_id: String) -> Result<OperationResponse, String> {
    let args = vec!["chat", "delete", &session_id, "--json"];

    let response: OperationResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Chat delete CLI failed: {}", e);
            return Err(format!("Chat delete failed: {}", e));
        }
    };

    log::info!("Deleted chat session '{}'", session_id);

    Ok(response)
}

/// Create a new chat session.
///
/// Returns a new session ID.
#[command]
pub async fn create_chat_session() -> Result<String, String> {
    let session_id = Uuid::new_v4().to_string();
    log::info!("Created new chat session: {}", session_id);
    Ok(session_id)
}
