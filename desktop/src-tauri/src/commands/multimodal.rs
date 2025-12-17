//! Multimodal IPC commands for audio/image/document processing.
//!
//! Step 08: Frontend Intelligence Integration - Phase 1
//!
//! Research Foundation:
//! - MM-HELIX: Multi-modal hybrid intelligent system
//! - Youtu-GraphRAG: Multi-modal information integration
//!
//! Bridges frontend multimodal input to backend processing services:
//! - Voice transcription via Whisper V3 (Ollama 10-100x faster, or HuggingFace)
//! - Image OCR via DeepSeek-OCR (>98% accuracy) with Tesseract fallback
//! - Document processing via existing normalization pipeline

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde::{Deserialize, Serialize};
use std::io::Write;
use tauri::command;
use tempfile::NamedTempFile;

/// Timestamped segment from transcription.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptSegment {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f64,
}

/// Response from voice transcription.
///
/// Research Foundation:
/// - Whisper V3: SOTA speech recognition with timestamps
/// - MM-HELIX: Temporal segment extraction for knowledge integration
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscribeResponse {
    pub success: bool,
    pub text: String,
    pub language: String,
    pub confidence: f64,
    pub segments: Vec<TranscriptSegment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Text region from OCR with bounding box.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrRegion {
    pub text: String,
    pub bbox: Vec<f64>,
    pub confidence: f64,
    #[serde(rename = "type")]
    pub region_type: String,
}

/// Layout information from OCR.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrLayout {
    pub regions: Vec<OcrRegion>,
}

/// Response from image OCR.
///
/// Research Foundation:
/// - DeepSeek-OCR: SOTA document understanding with layout preservation
/// - Youtu-GraphRAG: Visual information integration
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrResponse {
    pub success: bool,
    pub text: String,
    pub confidence: f64,
    pub layout: OcrLayout,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Response from document processing.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentResponse {
    pub success: bool,
    pub filename: String,
    pub text: String,
    pub summary: String,
    #[serde(rename = "type")]
    pub doc_type: String,
    pub word_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub element_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Backend status for multimodal services.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WhisperStatus {
    pub available: bool,
    pub backend: String,
    pub models: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrStatus {
    pub available: bool,
    pub backend: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MultimodalStatusResponse {
    pub success: bool,
    pub whisper: WhisperStatus,
    pub ocr: OcrStatus,
}

/// Transcribe voice audio to text using Whisper V3.
///
/// Step 08: Frontend Intelligence Integration - Phase 1
///
/// Research Foundation:
/// - Whisper V3: SOTA ASR with 98+ languages and timestamps
/// - Uses Ollama (10-100x faster) or HuggingFace fallback
///
/// Calls: `futurnal multimodal transcribe - --json` (with base64 stdin)
#[command]
pub async fn transcribe_voice(
    audio_data: Vec<u8>,
    language: Option<String>,
) -> Result<TranscribeResponse, String> {
    // Write audio data to temp file
    let temp_file = NamedTempFile::new()
        .map_err(|e| format!("Failed to create temp file: {}", e))?;

    std::fs::write(temp_file.path(), &audio_data)
        .map_err(|e| format!("Failed to write audio data: {}", e))?;

    let file_path = temp_file.path().to_string_lossy().to_string();

    // Build CLI arguments
    let mut args = vec![
        "multimodal",
        "transcribe",
        &file_path,
        "--json",
    ];

    let lang_str;
    if let Some(ref lang) = language {
        lang_str = lang.clone();
        args.push("--language");
        args.push(&lang_str);
    }

    // Execute Python CLI with longer timeout (first run downloads Whisper model)
    let response: TranscribeResponse = match crate::python::execute_cli_with_timeout(&args, 180).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Voice transcription CLI failed: {}", e);
            return Ok(TranscribeResponse {
                success: false,
                text: String::new(),
                language: language.unwrap_or_else(|| "unknown".to_string()),
                confidence: 0.0,
                segments: vec![],
                error: Some(format!("Transcription failed: {}", e)),
            });
        }
    };

    log::info!(
        "Voice transcription complete: {} chars, language: {}, confidence: {:.2}",
        response.text.len(),
        response.language,
        response.confidence
    );

    Ok(response)
}

/// Extract text from image using DeepSeek-OCR.
///
/// Step 08: Frontend Intelligence Integration - Phase 1
///
/// Research Foundation:
/// - DeepSeek-OCR: SOTA accuracy (>98%) with layout preservation
/// - Uses Tesseract as fallback
///
/// Calls: `futurnal multimodal ocr - --json` (with base64 stdin)
#[command]
pub async fn analyze_image(
    image_data: Vec<u8>,
    preserve_layout: Option<bool>,
) -> Result<OcrResponse, String> {
    // Write image data to temp file
    let temp_file = NamedTempFile::new()
        .map_err(|e| format!("Failed to create temp file: {}", e))?;

    std::fs::write(temp_file.path(), &image_data)
        .map_err(|e| format!("Failed to write image data: {}", e))?;

    let file_path = temp_file.path().to_string_lossy().to_string();

    // Build CLI arguments
    let mut args = vec![
        "multimodal",
        "ocr",
        &file_path,
        "--json",
    ];

    if preserve_layout.unwrap_or(true) {
        args.push("--preserve-layout");
    } else {
        args.push("--no-layout");
    }

    // Execute Python CLI
    let response: OcrResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Image OCR CLI failed: {}", e);
            return Ok(OcrResponse {
                success: false,
                text: String::new(),
                confidence: 0.0,
                layout: OcrLayout { regions: vec![] },
                error: Some(format!("OCR failed: {}", e)),
            });
        }
    };

    log::info!(
        "Image OCR complete: {} chars, {} regions, confidence: {:.2}",
        response.text.len(),
        response.layout.regions.len(),
        response.confidence
    );

    Ok(response)
}

/// Process a document through the normalization pipeline.
///
/// Step 08: Frontend Intelligence Integration - Phase 1
///
/// Handles various document types: PDF, DOCX, TXT, MD, HTML, etc.
///
/// Calls: `futurnal multimodal process - --filename <name> --json`
#[command]
pub async fn process_document(
    file_data: Vec<u8>,
    filename: String,
) -> Result<DocumentResponse, String> {
    // Write document data to temp file with proper extension
    let extension = std::path::Path::new(&filename)
        .extension()
        .map(|e| e.to_string_lossy().to_string())
        .unwrap_or_else(|| "txt".to_string());

    let temp_file = tempfile::Builder::new()
        .suffix(&format!(".{}", extension))
        .tempfile()
        .map_err(|e| format!("Failed to create temp file: {}", e))?;

    std::fs::write(temp_file.path(), &file_data)
        .map_err(|e| format!("Failed to write document data: {}", e))?;

    let file_path = temp_file.path().to_string_lossy().to_string();

    // Build CLI arguments
    let args = vec![
        "multimodal",
        "process",
        &file_path,
        "--filename",
        &filename,
        "--json",
    ];

    // Execute Python CLI with longer timeout for PDF processing
    let response: DocumentResponse = match crate::python::execute_cli_with_timeout(&args, 120).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Document processing CLI failed: {}", e);
            return Ok(DocumentResponse {
                success: false,
                filename: filename.clone(),
                text: String::new(),
                summary: String::new(),
                doc_type: extension,
                word_count: 0,
                element_count: None,
                error: Some(format!("Document processing failed: {}", e)),
            });
        }
    };

    log::info!(
        "Document processing complete: {} ({} words)",
        response.filename,
        response.word_count
    );

    Ok(response)
}

/// Response from image description using vision model.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DescribeImageResponse {
    pub success: bool,
    pub description: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Describe an image using a vision-capable LLM.
///
/// Step 08: Frontend Intelligence Integration - Phase 1
///
/// Uses Ollama with vision models (LLaVA, etc.) to understand
/// images that don't contain extractable text.
///
/// Calls: `futurnal multimodal describe <file> --model <model> --json`
#[command]
pub async fn describe_image(
    image_data: Vec<u8>,
    model: Option<String>,
    prompt: Option<String>,
) -> Result<DescribeImageResponse, String> {
    // Write image data to temp file
    let temp_file = NamedTempFile::new()
        .map_err(|e| format!("Failed to create temp file: {}", e))?;

    std::fs::write(temp_file.path(), &image_data)
        .map_err(|e| format!("Failed to write image data: {}", e))?;

    let file_path = temp_file.path().to_string_lossy().to_string();

    // Build CLI arguments
    let model_str = model.unwrap_or_else(|| "llava:7b".to_string());
    let mut args = vec![
        "multimodal",
        "describe",
        &file_path,
        "--model",
        &model_str,
        "--json",
    ];

    let prompt_str;
    if let Some(ref p) = prompt {
        prompt_str = p.clone();
        args.push("--prompt");
        args.push(&prompt_str);
    }

    // Execute Python CLI
    let response: DescribeImageResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Image description CLI failed: {}", e);
            return Ok(DescribeImageResponse {
                success: false,
                description: String::new(),
                model: model_str,
                image_type: None,
                error: Some(format!("Image description failed: {}", e)),
            });
        }
    };

    log::info!(
        "Image description complete: {} chars using {}",
        response.description.len(),
        response.model
    );

    Ok(response)
}

/// Vision status for multimodal services.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VisionStatus {
    pub available: bool,
    pub models: Vec<String>,
}

/// Check availability of multimodal backends.
///
/// Returns status of:
/// - Whisper V3 (Ollama / HuggingFace)
/// - DeepSeek-OCR / Tesseract
/// - Vision models (LLaVA, etc.)
///
/// Calls: `futurnal multimodal status --json`
#[command]
pub async fn get_multimodal_status() -> Result<MultimodalStatusResponse, String> {
    let args = vec!["multimodal", "status", "--json"];

    let response: MultimodalStatusResponse = match crate::python::execute_cli(&args).await {
        Ok(resp) => resp,
        Err(e) => {
            log::warn!("Multimodal status CLI failed: {}", e);
            return Ok(MultimodalStatusResponse {
                success: false,
                whisper: WhisperStatus {
                    available: false,
                    backend: "unknown".to_string(),
                    models: vec![],
                },
                ocr: OcrStatus {
                    available: false,
                    backend: "unknown".to_string(),
                },
            });
        }
    };

    log::info!(
        "Multimodal status: Whisper={} ({}), OCR={} ({})",
        response.whisper.available,
        response.whisper.backend,
        response.ocr.available,
        response.ocr.backend
    );

    Ok(response)
}
