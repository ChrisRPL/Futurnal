/**
 * Futurnal Desktop Shell - Multimodal API Client
 *
 * Step 08: Frontend Intelligence Integration - Phase 1
 *
 * Research Foundation:
 * - MM-HELIX: Multi-modal hybrid intelligent system
 * - Youtu-GraphRAG: Multi-modal information integration
 *
 * Provides typed wrappers for multimodal processing:
 * - Voice transcription via Whisper V3 (Ollama 10-100x faster, or HuggingFace)
 * - Image OCR via DeepSeek-OCR (>98% accuracy) with Tesseract fallback
 * - Document processing via existing normalization pipeline
 */

import { invoke } from '@tauri-apps/api/core';

/**
 * API Error class for handling Tauri IPC errors.
 */
class ApiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Default timeout for multimodal processing (longer than regular API calls).
 */
const DEFAULT_TIMEOUT_MS = 120000; // 2 minutes for audio/image processing

/**
 * Wrapper around invoke with timeout handling.
 */
async function invokeWithTimeout<T>(
  command: string,
  args?: Record<string, unknown>,
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<T> {
  const timeout = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new ApiError(`Request timeout: ${command}`)), timeoutMs);
  });

  try {
    const result = await Promise.race([invoke<T>(command, args), timeout]);
    return result;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (typeof error === 'string') throw new ApiError(error);
    throw new ApiError(String(error));
  }
}

// ============================================================================
// Types
// ============================================================================

/**
 * Timestamped segment from transcription.
 * Useful for linking audio segments to knowledge graph entities.
 */
export interface TranscriptSegment {
  text: string;
  start: number; // seconds
  end: number; // seconds
  confidence: number;
}

/**
 * Response from voice transcription.
 *
 * Research Foundation:
 * - Whisper V3: SOTA ASR with 98+ languages and temporal segments
 */
export interface TranscriptionResult {
  success: boolean;
  text: string;
  language: string;
  confidence: number;
  segments: TranscriptSegment[];
  error?: string;
}

/**
 * Text region from OCR with bounding box.
 * Useful for document layout understanding.
 */
export interface OcrRegion {
  text: string;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  confidence: number;
  type: string; // paragraph, heading, table, list, etc.
}

/**
 * Layout information from OCR.
 */
export interface OcrLayout {
  regions: OcrRegion[];
}

/**
 * Response from image OCR.
 *
 * Research Foundation:
 * - DeepSeek-OCR: SOTA document understanding (>98% accuracy)
 */
export interface OcrResult {
  success: boolean;
  text: string;
  confidence: number;
  layout: OcrLayout;
  error?: string;
}

/**
 * Response from document processing.
 */
export interface DocumentResult {
  success: boolean;
  filename: string;
  text: string;
  summary: string;
  type: string;
  wordCount: number;
  elementCount?: number;
  error?: string;
}

/**
 * Response from image description using vision model.
 *
 * Research Foundation:
 * - LLaVA: Vision-language model for image understanding
 * - Used when OCR finds no text (photos, diagrams, visual content)
 */
export interface ImageDescriptionResult {
  success: boolean;
  description: string;
  model: string;
  imageType?: string;
  error?: string;
}

/**
 * Backend status for multimodal services.
 */
export interface MultimodalStatus {
  success: boolean;
  whisper: {
    available: boolean;
    backend: 'ollama' | 'huggingface' | 'unknown';
    models: string[];
  };
  ocr: {
    available: boolean;
    backend: 'deepseek' | 'tesseract' | 'unknown';
  };
}

/**
 * Request for recording document learning.
 *
 * Research Foundation:
 * - SEAgent: Experiential learning loop
 * - Training-Free GRPO: Token priors for learning
 */
export interface RecordDocumentRequest {
  content: string;
  source?: string;
  contentType?: string;
  success?: boolean;
  qualityScore?: number;
  entityTypes?: string[];
  relationTypes?: string[];
}

/**
 * Response from recording document learning.
 */
export interface RecordDocumentResult {
  success: boolean;
  documentId: string;
  qualityScore: number;
  totalDocuments: number;
  overallSuccessRate: number;
  entityPriors: number;
  relationPriors: number;
  error?: string;
}

// ============================================================================
// Multimodal API
// ============================================================================

/**
 * Multimodal API for voice, image, and document processing.
 *
 * Step 08: Frontend Intelligence Integration - Phase 1
 *
 * Backend services (already implemented):
 * - Voice: Whisper V3 via Ollama (10-100x faster) or HuggingFace
 * - Images: DeepSeek-OCR (>98% accuracy) with Tesseract fallback
 * - Documents: Existing normalization pipeline
 */
export const multimodalApi = {
  /**
   * Transcribe voice audio to text using Whisper V3.
   *
   * Uses Ollama Whisper (10-100x faster) with HuggingFace fallback.
   *
   * @param audioData - Raw audio bytes (WAV, MP3, M4A, etc.)
   * @param language - Optional language code for forced detection (e.g., "en", "pl")
   * @returns TranscriptionResult with text, segments, and confidence
   *
   * @example
   * ```typescript
   * const blob = await recorder.stop();
   * const arrayBuffer = await blob.arrayBuffer();
   * const result = await multimodalApi.transcribeVoice(new Uint8Array(arrayBuffer));
   * console.log(result.text);
   * ```
   */
  async transcribeVoice(
    audioData: Uint8Array,
    language?: string
  ): Promise<TranscriptionResult> {
    try {
      const result = await invokeWithTimeout<TranscriptionResult>(
        'transcribe_voice',
        {
          audioData: Array.from(audioData),
          language,
        },
        180000 // 3 minutes for long audio
      );
      return result;
    } catch (error) {
      console.error('[Multimodal API] Voice transcription error:', error);
      return {
        success: false,
        text: '',
        language: language || 'unknown',
        confidence: 0,
        segments: [],
        error: String(error),
      };
    }
  },

  /**
   * Extract text from image using DeepSeek-OCR.
   *
   * Uses DeepSeek-OCR (>98% accuracy) with Tesseract fallback.
   *
   * @param imageData - Raw image bytes (PNG, JPG, etc.)
   * @param preserveLayout - Whether to preserve document layout (default: true)
   * @returns OcrResult with text, layout regions, and confidence
   *
   * @example
   * ```typescript
   * const file = event.target.files[0];
   * const arrayBuffer = await file.arrayBuffer();
   * const result = await multimodalApi.analyzeImage(new Uint8Array(arrayBuffer));
   * console.log(result.text);
   * // Access layout regions for structured extraction
   * result.layout.regions.forEach(region => {
   *   console.log(`${region.type}: ${region.text}`);
   * });
   * ```
   */
  async analyzeImage(
    imageData: Uint8Array,
    preserveLayout = true
  ): Promise<OcrResult> {
    try {
      const result = await invokeWithTimeout<OcrResult>(
        'analyze_image',
        {
          imageData: Array.from(imageData),
          preserveLayout,
        },
        120000 // 2 minutes for large images
      );
      return result;
    } catch (error) {
      console.error('[Multimodal API] Image OCR error:', error);
      return {
        success: false,
        text: '',
        confidence: 0,
        layout: { regions: [] },
        error: String(error),
      };
    }
  },

  /**
   * Process a document through the normalization pipeline.
   *
   * Handles various document types: PDF, DOCX, TXT, MD, HTML, etc.
   *
   * @param fileData - Raw file bytes
   * @param filename - Original filename (used for type detection)
   * @returns DocumentResult with text, summary, and metadata
   *
   * @example
   * ```typescript
   * const file = event.target.files[0];
   * const arrayBuffer = await file.arrayBuffer();
   * const result = await multimodalApi.processDocument(
   *   new Uint8Array(arrayBuffer),
   *   file.name
   * );
   * console.log(`${result.filename}: ${result.wordCount} words`);
   * ```
   */
  async processDocument(
    fileData: Uint8Array,
    filename: string
  ): Promise<DocumentResult> {
    try {
      const result = await invokeWithTimeout<DocumentResult>(
        'process_document',
        {
          fileData: Array.from(fileData),
          filename,
        },
        180000 // 3 minutes for large documents
      );
      return result;
    } catch (error) {
      console.error('[Multimodal API] Document processing error:', error);
      return {
        success: false,
        filename,
        text: '',
        summary: '',
        type: filename.split('.').pop() || 'unknown',
        wordCount: 0,
        error: String(error),
      };
    }
  },

  /**
   * Describe an image using a vision-capable LLM.
   *
   * Uses Ollama with vision models (LLaVA, etc.) to understand
   * images that don't contain extractable text.
   *
   * @param imageData - Raw image bytes (PNG, JPG, etc.)
   * @param model - Vision model to use (default: llava:7b)
   * @param prompt - Custom prompt for description
   * @returns ImageDescriptionResult with description and model info
   *
   * @example
   * ```typescript
   * const file = event.target.files[0];
   * const arrayBuffer = await file.arrayBuffer();
   * const result = await multimodalApi.describeImage(new Uint8Array(arrayBuffer));
   * console.log(result.description);
   * ```
   */
  async describeImage(
    imageData: Uint8Array,
    model?: string,
    prompt?: string
  ): Promise<ImageDescriptionResult> {
    try {
      const result = await invokeWithTimeout<ImageDescriptionResult>(
        'describe_image',
        {
          imageData: Array.from(imageData),
          model,
          prompt,
        },
        180000 // 3 minutes for vision model inference
      );
      return result;
    } catch (error) {
      console.error('[Multimodal API] Image description error:', error);
      return {
        success: false,
        description: '',
        model: model || 'llava:7b',
        error: String(error),
      };
    }
  },

  /**
   * Check availability of multimodal backends.
   *
   * @returns Status of Whisper and OCR backends
   *
   * @example
   * ```typescript
   * const status = await multimodalApi.getStatus();
   * if (status.whisper.available) {
   *   console.log(`Whisper using ${status.whisper.backend}`);
   * }
   * if (status.ocr.available) {
   *   console.log(`OCR using ${status.ocr.backend}`);
   * }
   * ```
   */
  async getStatus(): Promise<MultimodalStatus> {
    try {
      const result = await invokeWithTimeout<MultimodalStatus>(
        'get_multimodal_status',
        undefined,
        10000 // Quick status check
      );
      return result;
    } catch (error) {
      console.error('[Multimodal API] Status check error:', error);
      return {
        success: false,
        whisper: {
          available: false,
          backend: 'unknown',
          models: [],
        },
        ocr: {
          available: false,
          backend: 'unknown',
        },
      };
    }
  },

  /**
   * Record document processing to experiential learning pipeline.
   *
   * Updates learning state and token priors for continuous improvement.
   * Called after processing chat attachments.
   *
   * Research Foundation:
   * - SEAgent (2508.04700v2): Experiential learning loop
   * - Training-Free GRPO (2510.08191v1): Token priors
   *
   * @param request - Document recording request
   * @returns RecordDocumentResult with updated learning stats
   *
   * @example
   * ```typescript
   * const result = await multimodalApi.recordDocumentLearning({
   *   content: extractedText,
   *   contentType: 'image',
   *   success: true,
   *   entityTypes: ['Image', 'Document'],
   * });
   * console.log(`Total docs: ${result.totalDocuments}`);
   * ```
   */
  async recordDocumentLearning(
    request: RecordDocumentRequest
  ): Promise<RecordDocumentResult> {
    try {
      const result = await invokeWithTimeout<RecordDocumentResult>(
        'record_document_learning',
        { request },
        30000 // 30 seconds should be enough
      );
      return result;
    } catch (error) {
      console.error('[Multimodal API] Record document learning error:', error);
      return {
        success: false,
        documentId: '',
        qualityScore: 0,
        totalDocuments: 0,
        overallSuccessRate: 0,
        entityPriors: 0,
        relationPriors: 0,
        error: String(error),
      };
    }
  },
};

export default multimodalApi;
