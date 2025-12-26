/**
 * ChatInterface - Conversational interface to Personal Knowledge Graph
 *
 * Step 03: Chat Interface & Conversational AI
 *
 * Research Foundation:
 * - ProPerSim (2509.21730v1): Multi-turn context, session management
 * - Causal-Copilot (2504.13263v2): Natural language exploration, confidence
 *
 * Features:
 * - Multi-modal input (text, voice, images, files)
 * - Message bubbles with avatars and actions
 * - Typing indicator animation
 * - Confidence scoring display
 * - Source citations and entity references
 *
 * Styling: Per frontend-design.mdc - monochrome, dark-mode first
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import { MessageSquare, X, Link2, Trash2, FileText, Loader2, Globe, BookOpen, Zap, ExternalLink, CheckCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useChatStore } from '@/stores/chatStore';
import { multimodalApi } from '@/lib/multimodalApi';
import {
  papersApi,
  researchApi,
  type AgenticSearchResponse,
  type ScoredPaper,
  type WebSearchResponse,
  type DeepResearchResponse,
  type QuickSearchResponse,
} from '@/lib/api';
import { ChatBubble } from './ChatBubble';
import { MessageLoading } from './MessageLoading';
import { ChatInput, type Attachment } from './ChatInput';
import { ChatModelSelector } from './ChatModelSelector';
import { PaperSearchResults, type SearchResult } from './PaperSearchResults';

interface ChatInterfaceProps {
  /** Session ID to use (creates new if not provided) */
  sessionId?: string;
  /** Entity ID for "Ask about this" feature */
  contextEntityId?: string;
  /** Callback when an entity reference is clicked */
  onEntityClick?: (entityId: string) => void;
  /** Callback to close the chat panel */
  onClose?: () => void;
  /** Enable voice input */
  enableVoice?: boolean;
  /** Enable image upload */
  enableImages?: boolean;
  /** Enable file attachments */
  enableFiles?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export function ChatInterface({
  sessionId,
  contextEntityId,
  onEntityClick,
  onClose,
  enableVoice = true,
  enableImages = true,
  enableFiles = true,
  className,
}: ChatInterfaceProps) {
  const [input, setInput] = useState('');
  const [processingStatus, setProcessingStatus] = useState<string | null>(null);
  const [paperSearchResults, setPaperSearchResults] = useState<SearchResult | null>(null);
  const [agenticSearchResults, setAgenticSearchResults] = useState<AgenticSearchResponse | null>(null);
  const [isPaperSearching, setIsPaperSearching] = useState(false);
  const [searchProgress, setSearchProgress] = useState<string>('');
  // Research state
  const [webSearchResult, setWebSearchResult] = useState<WebSearchResponse | null>(null);
  const [deepResearchResult, setDeepResearchResult] = useState<DeepResearchResponse | null>(null);
  const [quickSearchResult, setQuickSearchResult] = useState<QuickSearchResponse | null>(null);
  const [isResearching, setIsResearching] = useState(false);
  const [researchProgress, setResearchProgress] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  const {
    currentSessionId,
    messages,
    isLoading,
    error,
    contextEntityId: storeContextEntity,
    createSession,
    setCurrentSession,
    sendMessage,
    setContextEntity,
    clearError,
    clearSession,
  } = useChatStore();

  // Initialize session on mount
  useEffect(() => {
    const initSession = async () => {
      if (sessionId && sessionId !== currentSessionId) {
        await setCurrentSession(sessionId);
      } else if (!currentSessionId) {
        await createSession();
      }
    };
    initSession();
  }, [sessionId, currentSessionId, setCurrentSession, createSession]);

  // Set context entity if provided
  useEffect(() => {
    if (contextEntityId && contextEntityId !== storeContextEntity) {
      setContextEntity(contextEntityId);
    }
  }, [contextEntityId, storeContextEntity, setContextEntity]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Clear error after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(clearError, 5000);
      return () => clearTimeout(timer);
    }
  }, [error, clearError]);

  // Handle send message
  // Step 08: Frontend Intelligence Integration - DeepSeek-OCR and document processing
  // UX: Show attachment chips (like ChatGPT), send extracted content as context to AI
  const handleSubmit = useCallback(
    async (text: string, attachments?: Attachment[]) => {
      if (!text.trim() && (!attachments || attachments.length === 0)) return;
      if (isLoading || processingStatus) return;

      // Clear input immediately for responsive UX
      setInput('');

      // If no attachments, send immediately
      if (!attachments || attachments.length === 0) {
        await sendMessage(text);
        return;
      }

      // Show processing status for attachments
      setProcessingStatus(`Analyzing ${attachments.length} attachment${attachments.length > 1 ? 's' : ''}...`);
      console.log('[Chat] Processing attachments:', attachments.length);

      // Build context from attachments (sent to AI but not shown in UI)
      const extractedContents: string[] = [];

      // Process attachments with multimodal API
      for (let i = 0; i < attachments.length; i++) {
        const attachment = attachments[i];
        setProcessingStatus(`Processing ${attachment.name} (${i + 1}/${attachments.length})...`);

        try {
          const arrayBuffer = await attachment.file.arrayBuffer();
          const data = new Uint8Array(arrayBuffer);

          if (attachment.type === 'image') {
            // Process image with BOTH OCR and vision model for comprehensive understanding
            const contentParts: string[] = [];

            // 1. OCR for text extraction
            console.log('[Chat] Analyzing image with OCR:', attachment.name);
            const ocrResult = await multimodalApi.analyzeImage(data);

            if (ocrResult.success && ocrResult.text.trim()) {
              contentParts.push(`Text extracted:\n${ocrResult.text.trim()}`);
              console.log('[Chat] OCR extracted:', ocrResult.text.length, 'chars');
            }

            // 2. Vision model for visual understanding (always run)
            console.log('[Chat] Analyzing image with vision model:', attachment.name);
            setProcessingStatus(`Analyzing image content (${i + 1}/${attachments.length})...`);

            const visionResult = await multimodalApi.describeImage(data);

            if (visionResult.success && visionResult.description.trim()) {
              contentParts.push(`Visual description:\n${visionResult.description.trim()}`);
              console.log('[Chat] Vision model described image:', visionResult.description.length, 'chars');
            } else {
              console.log('[Chat] Vision model unavailable or failed:', visionResult.error);
            }

            // Combine results
            if (contentParts.length > 0) {
              extractedContents.push(`[Image: ${attachment.name}]\n${contentParts.join('\n\n')}`);

              // Record to learning pipeline
              multimodalApi.recordDocumentLearning({
                content: contentParts.join('\n\n'),
                source: 'chat',
                contentType: 'image',
                success: true,
                entityTypes: ['Image', 'Document'],
              }).then(result => {
                if (result.success) {
                  console.log('[Chat] Learning recorded for image:', result.documentId);
                }
              }).catch(e => console.warn('[Chat] Learning record failed:', e));
            } else {
              extractedContents.push(`[Image: ${attachment.name}] (could not extract text or visual description)`);
            }
          } else {
            // Process document through normalization pipeline
            console.log('[Chat] Processing document:', attachment.name);
            const docResult = await multimodalApi.processDocument(data, attachment.name);

            if (docResult.success && docResult.text.trim()) {
              extractedContents.push(`[Document: ${attachment.name}]\n${docResult.text}`);
              console.log('[Chat] Document processed:', docResult.wordCount, 'words');

              // Record to learning pipeline
              multimodalApi.recordDocumentLearning({
                content: docResult.text,
                source: 'chat',
                contentType: 'document',
                success: true,
                entityTypes: ['Document'],
              }).then(result => {
                if (result.success) {
                  console.log('[Chat] Learning recorded for document:', result.documentId);
                }
              }).catch(e => console.warn('[Chat] Learning record failed:', e));
            } else if (docResult.error) {
              extractedContents.push(`[Document: ${attachment.name}] (processing failed: ${docResult.error})`);
              console.warn('[Chat] Document processing error:', docResult.error);
            }
          }
        } catch (err) {
          console.error('[Chat] Attachment processing failed:', attachment.name, err);
          extractedContents.push(`[${attachment.type === 'image' ? 'Image' : 'Document'}: ${attachment.name}] (failed to process)`);

          // Record failure to learning pipeline
          multimodalApi.recordDocumentLearning({
            content: '',
            source: 'chat',
            contentType: attachment.type === 'image' ? 'image' : 'document',
            success: false,
          }).catch(e => console.warn('[Chat] Learning record failed:', e));
        }
      }

      // Build attachment metadata for display
      const chatAttachments = attachments.map(a => ({
        id: a.id,
        type: a.type as 'image' | 'document',
        name: a.name,
        preview: a.preview,
        status: 'success' as const,
      }));

      // Build hidden context for AI
      const hiddenContext = extractedContents.length > 0
        ? extractedContents.join('\n\n')
        : undefined;

      setProcessingStatus(null);

      // Send message with attachments metadata and hidden context
      await sendMessage({
        content: text,
        hiddenContext,
        attachments: chatAttachments,
      });
    },
    [isLoading, processingStatus, sendMessage]
  );

  // Handle clear conversation
  const handleClear = async () => {
    if (confirm('Clear all messages in this conversation?')) {
      await clearSession();
    }
  };

  // Handle slash commands
  const handleSlashCommand = useCallback(
    async (action: string, args: string) => {
      console.log('[Chat] Slash command:', action, args);

      switch (action) {
        case 'save_insight':
          // Send a special message to save insight
          await sendMessage(`[SYSTEM: Save insight] ${args || 'Save the key insights from our conversation'}`);
          break;

        case 'paper_search':
          // Use agentic paper search for intelligent query handling
          if (!args.trim()) {
            await sendMessage('[Paper Search] Please provide a search query. Example: /paper causal inference');
            break;
          }

          setIsPaperSearching(true);
          setPaperSearchResults(null);
          setAgenticSearchResults(null);
          setSearchProgress('Analyzing query...');

          try {
            console.log('[Chat] Agentic paper search for:', args);

            // Use agentic search for intelligent multi-strategy search
            const response = await papersApi.agenticSearch(args.trim());

            // Check for API errors first
            if (!response.success) {
              await sendMessage(`Paper search failed: ${response.error || 'Unknown error'}`);
            } else if (response.papers.length > 0) {
              // Store the full agentic results (includes synthesis, suggestions)
              setAgenticSearchResults(response);

              // Convert to SearchResult format for PaperSearchResults component (camelCase)
              const searchResult: SearchResult = {
                query: response.query,
                total: response.totalEvaluated,
                papers: response.papers.map((sp: ScoredPaper) => ({
                  paperId: sp.paperId,
                  title: sp.title,
                  authors: sp.authors.map(a => ({ name: a.name, authorId: a.authorId })),
                  year: sp.year,
                  abstractText: sp.abstractText,
                  venue: sp.venue,
                  citationCount: sp.citationCount,
                  pdfUrl: sp.pdfUrl,
                  semanticScholarUrl: sp.sourceUrl,
                })),
                searchTimeMs: response.searchTimeMs,
              };
              setPaperSearchResults(searchResult);
            } else if (response.synthesis) {
              // No papers but we have synthesis (maybe suggestions)
              setAgenticSearchResults(response);
            } else {
              await sendMessage(`No papers found for "${args}". Try different keywords or a broader search term.`);
            }
          } catch (error) {
            console.error('[Chat] Agentic paper search failed:', error);
            await sendMessage(`Paper search failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
          } finally {
            setIsPaperSearching(false);
            setSearchProgress('');
          }
          break;

        case 'causal_analysis':
          // Send a special message for causal analysis
          await sendMessage(`[SYSTEM: Causal analysis] Analyze causal factors for: ${args}`);
          break;

        case 'show_insights':
          // Send a special message to show insights
          await sendMessage('[SYSTEM: Show insights] Show me the emergent insights from my data');
          break;

        case 'web_search':
          // Web search with autonomous browsing
          if (!args.trim()) {
            await sendMessage('[Web Search] Please provide a search query. Example: /web what is causal inference');
            break;
          }

          setIsResearching(true);
          setWebSearchResult(null);
          setDeepResearchResult(null);
          setQuickSearchResult(null);
          setResearchProgress('Searching the web...');

          try {
            console.log('[Chat] Web search for:', args);
            const response = await researchApi.webSearch({ query: args.trim() });

            if (response.success) {
              setWebSearchResult(response);
            } else {
              await sendMessage(`Web search failed: ${response.error || 'Unknown error'}`);
            }
          } catch (error) {
            console.error('[Chat] Web search failed:', error);
            await sendMessage(`Web search failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
          } finally {
            setIsResearching(false);
            setResearchProgress('');
          }
          break;

        case 'deep_research':
          // Deep research combining PKG and web
          if (!args.trim()) {
            await sendMessage('[Deep Research] Please provide a research topic. Example: /research impact of sleep on productivity');
            break;
          }

          setIsResearching(true);
          setWebSearchResult(null);
          setDeepResearchResult(null);
          setQuickSearchResult(null);
          setResearchProgress('Conducting deep research...');

          try {
            console.log('[Chat] Deep research for:', args);
            const response = await researchApi.deepResearch({
              query: args.trim(),
              depth: 'detailed',
            });

            if (response.success) {
              setDeepResearchResult(response);
            } else {
              await sendMessage(`Deep research failed: ${response.error || 'Unknown error'}`);
            }
          } catch (error) {
            console.error('[Chat] Deep research failed:', error);
            await sendMessage(`Deep research failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
          } finally {
            setIsResearching(false);
            setResearchProgress('');
          }
          break;

        case 'quick_search':
          // Quick web search without synthesis
          if (!args.trim()) {
            await sendMessage('[Quick Search] Please provide a search query. Example: /quick python async tutorial');
            break;
          }

          setIsResearching(true);
          setWebSearchResult(null);
          setDeepResearchResult(null);
          setQuickSearchResult(null);
          setResearchProgress('Searching...');

          try {
            console.log('[Chat] Quick search for:', args);
            const response = await researchApi.quickSearch(args.trim());

            if (response.success) {
              setQuickSearchResult(response);
            } else {
              await sendMessage(`Quick search failed: ${response.error || 'Unknown error'}`);
            }
          } catch (error) {
            console.error('[Chat] Quick search failed:', error);
            await sendMessage(`Quick search failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
          } finally {
            setIsResearching(false);
            setResearchProgress('');
          }
          break;

        default:
          console.log('[Chat] Unknown slash command action:', action);
      }
    },
    [sendMessage]
  );

  return (
    <div className={cn('flex flex-col h-full bg-black', className)}>
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-white/10">
        <MessageSquare className="h-4 w-4 text-white/60" />
        <span className="text-sm font-medium text-white/80">
          Chat with Knowledge
        </span>

        <div className="ml-auto flex items-center gap-2">
          {/* Model selector */}
          <ChatModelSelector />
          {/* Clear button */}
          {messages.length > 0 && (
            <button
              onClick={handleClear}
              className="p-1.5 text-white/40 hover:text-white/60 hover:bg-white/5 transition-colors"
              title="Clear conversation"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          )}

          {/* Close button */}
          {onClose && (
            <button
              onClick={onClose}
              className="p-1.5 text-white/40 hover:text-white/60 hover:bg-white/5 transition-colors"
              title="Close chat"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* Context indicator */}
      {(storeContextEntity || contextEntityId) && (
        <div className="px-4 py-2 border-b border-white/5 bg-white/[0.02]">
          <div className="flex items-center gap-2 text-xs text-white/60">
            <Link2 className="h-3 w-3" />
            <span>Discussing:</span>
            <span className="text-white/80 font-medium">
              {storeContextEntity || contextEntityId}
            </span>
            <button
              onClick={() => setContextEntity(null)}
              className="ml-auto text-white/40 hover:text-white/60 p-0.5"
              title="Clear context"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div className="px-4 py-2 bg-red-500/10 border-b border-red-500/20">
          <p className="text-xs text-red-400">{error}</p>
        </div>
      )}

      {/* Messages */}
      <div
        ref={messagesContainerRef}
        className="flex-1 overflow-y-auto p-4 space-y-4"
      >
        {/* Empty state */}
        {messages.length === 0 && !isLoading && (
          <div className="flex flex-col items-center justify-center h-full text-white/50">
            <MessageSquare className="h-12 w-12 mb-4 opacity-30" />
            <p className="text-sm font-medium">Start a conversation</p>
            <p className="text-xs mt-2 text-white/30 text-center max-w-[200px]">
              Ask questions about your personal knowledge graph
            </p>
            <div className="mt-6 flex flex-wrap gap-2 justify-center max-w-[280px]">
              {[
                'What do I know about...',
                'Summarize my notes on...',
                'Find connections between...',
              ].map((hint) => (
                <button
                  key={hint}
                  onClick={() => setInput(hint)}
                  className="text-xs px-2 py-1 bg-white/5 border border-white/10 text-white/50 hover:text-white/70 hover:bg-white/10 transition-colors"
                >
                  {hint}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Message list */}
        {messages.map((msg, i) => (
          <ChatBubble
            key={`${msg.timestamp}-${i}`}
            message={msg}
            onEntityClick={onEntityClick}
            sessionId={currentSessionId || undefined}
          />
        ))}

        {/* Processing attachments indicator */}
        {processingStatus && (
          <div className="flex items-center gap-3 px-4 py-3 bg-white/5 border border-white/10 animate-pulse">
            <div className="h-2 w-2 bg-white/60 rounded-full animate-ping" />
            <span className="text-sm text-white/70">{processingStatus}</span>
          </div>
        )}

        {/* Loading indicator */}
        {isLoading && <MessageLoading />}

        {/* Paper search loading indicator */}
        {isPaperSearching && (
          <div className="flex items-center gap-3 px-4 py-3 bg-white/5 border border-white/10 rounded-lg">
            <Loader2 className="h-4 w-4 text-white/60 animate-spin" />
            <div className="flex flex-col">
              <span className="text-sm text-white/70">Intelligent paper search in progress...</span>
              {searchProgress && (
                <span className="text-xs text-white/40">{searchProgress}</span>
              )}
            </div>
          </div>
        )}

        {/* Agentic paper search results */}
        {(paperSearchResults || agenticSearchResults) && (
          <div className="p-4 bg-white/[0.02] border border-white/10 rounded-lg space-y-4">
            {/* Header */}
            <div className="flex items-center gap-2 pb-2 border-b border-white/10">
              <FileText className="h-4 w-4 text-white/60" />
              <span className="text-sm font-medium text-white/80">
                Agentic Paper Search: "{agenticSearchResults?.query || paperSearchResults?.query}"
              </span>
              <button
                onClick={() => {
                  setPaperSearchResults(null);
                  setAgenticSearchResults(null);
                }}
                className="ml-auto p-1 text-white/40 hover:text-white/60 transition-colors"
                title="Dismiss results"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>

            {/* Synthesis - AI summary of findings */}
            {agenticSearchResults?.synthesis && (
              <div className="p-3 bg-white/5 rounded-lg border border-white/10">
                <div className="text-xs font-medium text-white/50 uppercase tracking-wider mb-2">
                  Synthesis
                </div>
                <p className="text-sm text-white/80 leading-relaxed">
                  {agenticSearchResults.synthesis}
                </p>
              </div>
            )}

            {/* Strategies tried */}
            {agenticSearchResults?.strategiesTried && agenticSearchResults.strategiesTried.length > 0 && (
              <div className="text-xs text-white/40">
                <span className="font-medium">Strategies tried: </span>
                {agenticSearchResults.strategiesTried.map(s => s.type).join(', ')}
                <span className="mx-2">•</span>
                <span>{agenticSearchResults.totalEvaluated} papers evaluated</span>
              </div>
            )}

            {/* Paper results */}
            {paperSearchResults && paperSearchResults.papers.length > 0 && (
              <PaperSearchResults
                results={paperSearchResults}
                onDownload={(papers) => {
                  console.log('[Chat] Papers added to KG:', papers.length);
                }}
              />
            )}

            {/* Suggestions */}
            {agenticSearchResults?.suggestions && agenticSearchResults.suggestions.length > 0 && (
              <div className="pt-3 border-t border-white/10">
                <div className="text-xs font-medium text-white/50 uppercase tracking-wider mb-2">
                  Try these searches
                </div>
                <div className="flex flex-wrap gap-2">
                  {agenticSearchResults.suggestions.map((suggestion, i) => {
                    // Extract search term from suggestion
                    const match = suggestion.match(/["']([^"']+)["']/i) ||
                                  suggestion.match(/search\s+for\s+(.+?)(?:\.|$)/i) ||
                                  suggestion.match(/try\s+(.+?)(?:\.|$)/i);
                    const searchTerm = match ? match[1] : suggestion.replace(/^(try|search|search for)\s*/i, '').trim();

                    return (
                      <button
                        key={i}
                        onClick={() => {
                          // Directly trigger a new paper search
                          handleSlashCommand('paper_search', searchTerm);
                        }}
                        className="px-2 py-1 text-xs bg-white/5 border border-white/10 rounded text-white/60 hover:text-white/80 hover:bg-white/10 transition-colors"
                        title={`Search for: ${searchTerm}`}
                      >
                        {searchTerm.length > 40 ? `${searchTerm.slice(0, 40)}...` : searchTerm}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Search time */}
            {agenticSearchResults?.searchTimeMs && (
              <div className="text-xs text-white/30 text-right">
                Search completed in {(agenticSearchResults.searchTimeMs / 1000).toFixed(1)}s
              </div>
            )}
          </div>
        )}

        {/* Research loading indicator */}
        {isResearching && (
          <div className="flex items-center gap-3 px-4 py-3 bg-white/5 border border-white/10 rounded-lg">
            <Loader2 className="h-4 w-4 text-white/60 animate-spin" />
            <div className="flex flex-col">
              <span className="text-sm text-white/70">{researchProgress || 'Researching...'}</span>
              <span className="text-xs text-white/40">This may take a moment</span>
            </div>
          </div>
        )}

        {/* Web Search Results */}
        {webSearchResult && (
          <div className="p-4 bg-white/[0.02] border border-white/10 rounded-lg space-y-4">
            {/* Header */}
            <div className="flex items-center gap-2 pb-2 border-b border-white/10">
              <Globe className="h-4 w-4 text-white/60" />
              <span className="text-sm font-medium text-white/80">
                Web Search: "{webSearchResult.query}"
              </span>
              <button
                onClick={() => setWebSearchResult(null)}
                className="ml-auto p-1 text-white/40 hover:text-white/60 transition-colors"
                title="Dismiss results"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>

            {/* Answer */}
            {webSearchResult.answer && (
              <div className="p-3 bg-white/5 rounded-lg border border-white/10">
                <p className="text-sm text-white/80 leading-relaxed whitespace-pre-wrap">
                  {webSearchResult.answer}
                </p>
              </div>
            )}

            {/* Sources */}
            {webSearchResult.sources.length > 0 && (
              <div>
                <div className="text-xs font-medium text-white/50 uppercase tracking-wider mb-2">
                  Sources ({webSearchResult.sources.length})
                </div>
                <div className="space-y-2">
                  {webSearchResult.sources.slice(0, 5).map((source, i) => (
                    <a
                      key={i}
                      href={source.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 p-2 bg-white/5 rounded border border-white/10 hover:bg-white/10 transition-colors"
                    >
                      <ExternalLink className="h-3 w-3 text-white/40 flex-shrink-0" />
                      <span className="text-xs text-white/70 truncate flex-1">{source.title || source.url}</span>
                      <span className={cn(
                        "text-[10px] px-1.5 py-0.5 rounded",
                        source.reliability === 'high' ? 'bg-emerald-500/20 text-emerald-400' :
                        source.reliability === 'medium' ? 'bg-amber-500/20 text-amber-400' :
                        'bg-white/10 text-white/40'
                      )}>
                        {source.reliability}
                      </span>
                    </a>
                  ))}
                </div>
              </div>
            )}

            {/* Confidence */}
            <div className="flex items-center gap-4 text-xs text-white/40">
              <span>Confidence: {Math.round(webSearchResult.confidence * 100)}%</span>
              <span>Pages visited: {webSearchResult.totalPages}</span>
              <span className="ml-auto">{(webSearchResult.searchTimeMs / 1000).toFixed(1)}s</span>
            </div>
          </div>
        )}

        {/* Deep Research Results */}
        {deepResearchResult && (
          <div className="p-4 bg-white/[0.02] border border-white/10 rounded-lg space-y-4">
            {/* Header */}
            <div className="flex items-center gap-2 pb-2 border-b border-white/10">
              <BookOpen className="h-4 w-4 text-white/60" />
              <span className="text-sm font-medium text-white/80">
                Deep Research: "{deepResearchResult.query}"
              </span>
              <button
                onClick={() => setDeepResearchResult(null)}
                className="ml-auto p-1 text-white/40 hover:text-white/60 transition-colors"
                title="Dismiss results"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>

            {/* Summary */}
            {deepResearchResult.summary && (
              <div className="p-3 bg-white/5 rounded-lg border border-white/10">
                <div className="text-xs font-medium text-white/50 uppercase tracking-wider mb-2">
                  Summary
                </div>
                <p className="text-sm text-white/80 leading-relaxed whitespace-pre-wrap">
                  {deepResearchResult.summary}
                </p>
              </div>
            )}

            {/* Key Points */}
            {deepResearchResult.keyPoints.length > 0 && (
              <div>
                <div className="text-xs font-medium text-white/50 uppercase tracking-wider mb-2">
                  Key Points
                </div>
                <ul className="space-y-1.5">
                  {deepResearchResult.keyPoints.map((point, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-white/70">
                      <CheckCircle className="h-3.5 w-3.5 text-emerald-400/60 mt-0.5 flex-shrink-0" />
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Metadata */}
            <div className="flex items-center gap-4 text-xs text-white/40 pt-2 border-t border-white/10">
              <span>Depth: {deepResearchResult.depthUsed}</span>
              <span>Sources: {deepResearchResult.numSourcesConsulted}</span>
              <span>Confidence: {Math.round(deepResearchResult.confidence * 100)}%</span>
              <span className="ml-auto">{(deepResearchResult.researchTimeMs / 1000).toFixed(1)}s</span>
            </div>
          </div>
        )}

        {/* Quick Search Results */}
        {quickSearchResult && quickSearchResult.results.length > 0 && (
          <div className="p-4 bg-white/[0.02] border border-white/10 rounded-lg space-y-4">
            {/* Header */}
            <div className="flex items-center gap-2 pb-2 border-b border-white/10">
              <Zap className="h-4 w-4 text-white/60" />
              <span className="text-sm font-medium text-white/80">
                Quick Search: "{quickSearchResult.query}"
              </span>
              <span className="text-xs text-white/40">
                ({quickSearchResult.total} results)
              </span>
              <button
                onClick={() => setQuickSearchResult(null)}
                className="ml-auto p-1 text-white/40 hover:text-white/60 transition-colors"
                title="Dismiss results"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>

            {/* Results */}
            <div className="space-y-3">
              {quickSearchResult.results.map((result, i) => (
                <a
                  key={i}
                  href={result.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block p-3 bg-white/5 rounded border border-white/10 hover:bg-white/10 transition-colors"
                >
                  <div className="flex items-start gap-2">
                    <ExternalLink className="h-3.5 w-3.5 text-white/40 mt-0.5 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-white/80 truncate">
                        {result.title || 'Untitled'}
                      </div>
                      {result.snippet && (
                        <p className="text-xs text-white/50 mt-1 line-clamp-2">
                          {result.snippet}
                        </p>
                      )}
                      <div className="text-[10px] text-white/30 mt-1 truncate">
                        {result.url}
                      </div>
                    </div>
                  </div>
                </a>
              ))}
            </div>

            {/* Search time */}
            <div className="text-xs text-white/30 text-right">
              Search completed in {quickSearchResult.searchTimeMs}ms
            </div>
          </div>
        )}

        {/* Scroll anchor */}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-white/10">
        <ChatInput
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          onSlashCommand={handleSlashCommand}
          isLoading={isLoading || !!processingStatus}
          enableVoice={enableVoice}
          enableImages={enableImages}
          enableFiles={enableFiles}
          placeholder={
            processingStatus
              ? processingStatus
              : storeContextEntity
                ? `Ask about ${storeContextEntity}...`
                : 'Ask about your knowledge...'
          }
        />

        {/* Message count */}
        {messages.length > 0 && (
          <div className="mt-2 text-xs text-white/30 text-right">
            {messages.length} messages in this conversation
          </div>
        )}
      </div>
    </div>
  );
}
