/**
 * ChatInput - Multi-modal input component for chat
 *
 * Step 03: Chat Interface & Conversational AI
 *
 * Supports:
 * - Text input (default)
 * - Voice recording (for speech-to-text models)
 * - Image upload (for OCR/vision models)
 * - File attachments
 *
 * Styling: Per frontend-design.mdc - monochrome, dark-mode first
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import {
  Send,
  Mic,
  MicOff,
  Image,
  Paperclip,
  X,
  Loader2,
  FileText,
  FileImage,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { multimodalApi } from '@/lib/multimodalApi';

// Supported attachment types
type AttachmentType = 'image' | 'file';

interface Attachment {
  id: string;
  type: AttachmentType;
  name: string;
  file: File;
  preview?: string; // Base64 for images
}

interface ChatInputProps {
  /** Current text value */
  value: string;
  /** Text change handler */
  onChange: (value: string) => void;
  /** Submit handler (text + optional attachments) */
  onSubmit: (text: string, attachments?: Attachment[]) => void;
  /** Whether the chat is processing */
  isLoading?: boolean;
  /** Placeholder text */
  placeholder?: string;
  /** Enable voice input */
  enableVoice?: boolean;
  /** Enable image upload */
  enableImages?: boolean;
  /** Enable file attachments */
  enableFiles?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Voice recording hook
 * Note: In Tauri WebView, navigator.mediaDevices may not be available.
 * We check for availability and provide a graceful fallback.
 */
function useVoiceRecording() {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Assume supported if API exists - we'll check permission when user clicks
  const [isSupported, setIsSupported] = useState(
    typeof navigator !== 'undefined' &&
    !!navigator.mediaDevices &&
    !!navigator.mediaDevices.getUserMedia
  );
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError('Voice recording not available in this environment');
      setIsSupported(false);
      console.error('Voice recording error: navigator.mediaDevices.getUserMedia not available');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
      setError(null);
    } catch (err) {
      setError('Microphone access denied');
      setIsSupported(false);
      console.error('Voice recording error:', err);
    }
  }, []);

  const stopRecording = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      if (!mediaRecorderRef.current) {
        resolve(null);
        return;
      }

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        setIsRecording(false);

        // Stop all tracks
        mediaRecorderRef.current?.stream.getTracks().forEach((t) => t.stop());
        mediaRecorderRef.current = null;

        resolve(blob);
      };

      mediaRecorderRef.current.stop();
    });
  }, []);

  return {
    isRecording,
    isProcessing,
    isSupported,
    error,
    startRecording,
    stopRecording,
    setIsProcessing,
  };
}

/**
 * Attachment preview component
 */
function AttachmentPreview({
  attachment,
  onRemove,
}: {
  attachment: Attachment;
  onRemove: () => void;
}) {
  return (
    <div className="relative group flex items-center gap-2 px-2 py-1 bg-white/5 border border-white/10">
      {/* Icon */}
      {attachment.type === 'image' ? (
        attachment.preview ? (
          <img
            src={attachment.preview}
            alt={attachment.name}
            className="w-8 h-8 object-cover"
          />
        ) : (
          <FileImage className="w-4 h-4 text-white/60" />
        )
      ) : (
        <FileText className="w-4 h-4 text-white/60" />
      )}

      {/* Name */}
      <span className="text-xs text-white/70 truncate max-w-[100px]">
        {attachment.name}
      </span>

      {/* Remove button */}
      <button
        onClick={onRemove}
        className="p-0.5 text-white/40 hover:text-white/70 hover:bg-white/10 transition-colors"
        title="Remove"
      >
        <X className="w-3 h-3" />
      </button>
    </div>
  );
}

/**
 * Main ChatInput component
 */
export function ChatInput({
  value,
  onChange,
  onSubmit,
  isLoading = false,
  placeholder = 'Ask about your knowledge...',
  enableVoice = true,
  enableImages = true,
  enableFiles = true,
  className,
}: ChatInputProps) {
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { isRecording, isProcessing, isSupported: voiceSupported, error, startRecording, stopRecording, setIsProcessing } =
    useVoiceRecording();

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [value]);

  // Handle submit
  const handleSubmit = useCallback(() => {
    if ((!value.trim() && attachments.length === 0) || isLoading) return;

    onSubmit(value, attachments.length > 0 ? attachments : undefined);
    setAttachments([]);
  }, [value, attachments, isLoading, onSubmit]);

  // Handle key down
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Handle voice toggle
  // Step 08: Frontend Intelligence Integration - Whisper V3 transcription
  const handleVoiceToggle = async () => {
    if (isRecording) {
      const blob = await stopRecording();
      if (blob) {
        // Transcribe using Whisper V3 (Ollama 10-100x faster, or HuggingFace fallback)
        setIsProcessing(true);
        try {
          // Convert blob to Uint8Array for Tauri IPC
          const arrayBuffer = await blob.arrayBuffer();
          const audioData = new Uint8Array(arrayBuffer);

          console.log('[Voice] Transcribing audio:', audioData.length, 'bytes');

          const result = await multimodalApi.transcribeVoice(audioData);

          if (result.success && result.text) {
            // Append transcribed text to input (preserving existing text)
            const separator = value.trim() ? ' ' : '';
            onChange(value + separator + result.text);
            console.log('[Voice] Transcription complete:', result.text.length, 'chars, language:', result.language);
          } else if (result.error) {
            console.error('[Voice] Transcription error:', result.error);
          }
        } catch (err) {
          console.error('[Voice] Transcription failed:', err);
        } finally {
          setIsProcessing(false);
        }
      }
    } else {
      startRecording();
    }
  };

  // Handle image selection
  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    Array.from(files).forEach((file) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const attachment: Attachment = {
          id: `img-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          type: 'image',
          name: file.name,
          file,
          preview: reader.result as string,
        };
        setAttachments((prev) => [...prev, attachment]);
      };
      reader.readAsDataURL(file);
    });

    // Reset input
    if (imageInputRef.current) {
      imageInputRef.current.value = '';
    }
  };

  // Handle file selection
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    Array.from(files).forEach((file) => {
      const attachment: Attachment = {
        id: `file-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        type: 'file',
        name: file.name,
        file,
      };
      setAttachments((prev) => [...prev, attachment]);
    });

    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Remove attachment
  const removeAttachment = (id: string) => {
    setAttachments((prev) => prev.filter((a) => a.id !== id));
  };

  const hasContent = value.trim() || attachments.length > 0;

  return (
    <div className={cn('space-y-2', className)}>
      {/* Attachments preview */}
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-2 px-1">
          {attachments.map((attachment) => (
            <AttachmentPreview
              key={attachment.id}
              attachment={attachment}
              onRemove={() => removeAttachment(attachment.id)}
            />
          ))}
        </div>
      )}

      {/* Input row */}
      <div className="flex items-end gap-2">
        {/* Action buttons */}
        <div className="flex items-center gap-1 pb-2">
          {/* Voice button */}
          {enableVoice && (
            <button
              onClick={handleVoiceToggle}
              disabled={isProcessing || !voiceSupported}
              className={cn(
                'p-2 transition-colors',
                !voiceSupported
                  ? 'text-white/20 cursor-not-allowed'
                  : isRecording
                    ? 'text-red-400 bg-red-400/10 animate-pulse'
                    : 'text-white/40 hover:text-white/70 hover:bg-white/5'
              )}
              title={!voiceSupported ? 'Voice not available in Tauri' : isRecording ? 'Stop recording' : 'Start voice input'}
            >
              {isProcessing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : isRecording ? (
                <MicOff className="h-4 w-4" />
              ) : (
                <Mic className="h-4 w-4" />
              )}
            </button>
          )}

          {/* Image button */}
          {enableImages && (
            <>
              <input
                ref={imageInputRef}
                type="file"
                accept="image/*"
                multiple
                onChange={handleImageSelect}
                className="hidden"
              />
              <button
                onClick={() => imageInputRef.current?.click()}
                className="p-2 text-white/40 hover:text-white/70 hover:bg-white/5 transition-colors"
                title="Add image (OCR/Vision)"
              >
                <Image className="h-4 w-4" />
              </button>
            </>
          )}

          {/* File button */}
          {enableFiles && (
            <>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                onChange={handleFileSelect}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="p-2 text-white/40 hover:text-white/70 hover:bg-white/5 transition-colors"
                title="Attach file"
              >
                <Paperclip className="h-4 w-4" />
              </button>
            </>
          )}
        </div>

        {/* Text input */}
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={isRecording ? 'Recording...' : placeholder}
          disabled={isRecording}
          rows={1}
          className={cn(
            'flex-1 px-4 py-2',
            'bg-white/5 border border-white/10',
            'text-white text-sm placeholder-white/30',
            'focus:outline-none focus:border-white/20',
            'resize-none',
            'transition-colors',
            isRecording && 'opacity-50'
          )}
          style={{ minHeight: '40px', maxHeight: '120px' }}
        />

        {/* Send button */}
        <button
          onClick={handleSubmit}
          disabled={isLoading || !hasContent}
          className={cn(
            'p-2.5',
            'bg-white text-black',
            'hover:bg-white/90',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            'transition-all'
          )}
          title="Send message"
        >
          {isLoading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </button>
      </div>

      {/* Footer hints */}
      <div className="flex items-center justify-between text-xs text-white/30 px-1">
        <span>
          {isRecording
            ? 'Click mic to stop recording'
            : 'Enter to send, Shift+Enter for new line'}
        </span>
        {error && <span className="text-red-400">{error}</span>}
      </div>
    </div>
  );
}

// Export attachment type for use in other components
export type { Attachment, AttachmentType };
