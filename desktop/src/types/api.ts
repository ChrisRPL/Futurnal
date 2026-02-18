/**
 * Futurnal Desktop Shell - API Types
 *
 * TypeScript interfaces matching the Rust command structures.
 * These types ensure type safety between the frontend and backend.
 */

// ============================================================================
// Search Types
// ============================================================================

export interface SearchQuery {
  query: string;
  top_k?: number;
  filters?: SearchFilters;
}

export interface SearchFilters {
  entity_types?: string[];
  source_types?: string[];
  date_range?: DateRange;
  sources?: string[];
}

export interface DateRange {
  start?: string;
  end?: string;
}

export interface SearchResult {
  id: string;
  content: string;
  score: number;
  confidence: number;
  timestamp?: string;
  entity_type?: EntityType;
  source_type?: SourceType;
  source_confidence?: number;
  causal_chain?: CausalChain;
  metadata: Record<string, unknown>;
  /** Graph context from GraphRAG traversal (per GFM-RAG paper) */
  graph_context?: GraphContext;
  /** Vector similarity score from semantic search (0-1) */
  vector_score?: number;
  /** Graph traversal score from graph expansion (0-1) */
  graph_score?: number;
}

export type EntityType = 'Event' | 'Document' | 'Code' | 'Person' | 'Concept' | 'Email' | 'Mailbox' | 'Source' | 'Organization' | 'Entity';
export type SourceType = 'text' | 'ocr' | 'audio' | 'code';

export interface CausalChain {
  anchor: string;
  causes: string[];
  effects: string[];
}

/**
 * Graph traversal context for GraphRAG results.
 *
 * Per GFM-RAG paper (2502.01113v1):
 * - Shows "why" a result is relevant via graph connections
 * - Enables path visualization for user understanding
 * - Supports multi-hop reasoning explanation
 */
export interface GraphContext {
  /** Entities connected via graph traversal */
  relatedEntities: Array<{
    id: string;
    type: string;
    name?: string;
  }>;
  /** Relationships traversed to reach this result */
  relationships: Array<{
    type: string;
    source: string;
    target: string;
    confidence?: number;
  }>;
  /** Path from query entity to this result */
  pathToQuery: string[];
  /** Number of hops from seed entities */
  hopCount: number;
  /** Confidence score for the traversal path */
  pathConfidence: number;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  query_id: string;
  intent: QueryIntent;
  execution_time_ms: number;
}

/**
 * Search response with LLM-generated answer.
 *
 * Step 02: LLM Answer Generation
 * Research Foundation:
 * - CausalRAG (ACL 2025): Causal-aware generation
 * - LLM-Enhanced Symbolic (2501.01246v1): Hybrid approach
 */
export interface SearchWithAnswerResponse extends SearchResponse {
  /** LLM-generated synthesized answer */
  answer?: string;
  /** Sources used for answer generation */
  sources?: string[];
  /** Model used for answer generation */
  model?: string;
  /** Time taken for answer generation (ms) */
  generation_time_ms?: number;
}

/**
 * Available models for answer generation.
 * Per docs/LLM_MODEL_REGISTRY.md
 */
export interface AnswerModel {
  id: string;
  label: string;
  vram: string;
  hint: string;
}

export const ANSWER_MODELS: AnswerModel[] = [
  { id: 'phi3:mini', label: 'Phi-3 Mini', vram: '4GB', hint: 'Fast' },
  { id: 'llama3.1:8b-instruct-q4_0', label: 'Llama 3.1 8B', vram: '8GB', hint: 'Balanced' },
  { id: 'SpeakLeash/bielik-4.5b-v3.0-instruct:Q8_0', label: 'Bielik 4.5B', vram: '5GB', hint: 'Polish' },
  { id: 'qwen2.5:7b-instruct', label: 'Qwen 2.5 7B', vram: '8GB', hint: 'Quality' },
  { id: 'mistral:7b-instruct', label: 'Mistral 7B', vram: '8GB', hint: 'Reasoning' },
  { id: 'kimi-k2-thinking:cloud', label: 'Kimi K2 Thinking', vram: 'Cloud', hint: '1T MoE, Deep Reasoning' },
  { id: 'gpt-oss:20b', label: 'GPT-OSS 20B', vram: '12GB', hint: 'Local 20B' },
];

/**
 * Vision models for image understanding.
 * Used automatically when images without text are attached.
 */
export const VISION_MODELS: AnswerModel[] = [
  { id: 'llava:7b', label: 'LLaVA 7B', vram: '8GB', hint: 'Vision' },
  { id: 'llava:13b', label: 'LLaVA 13B', vram: '12GB', hint: 'Quality Vision' },
  { id: 'moondream:latest', label: 'Moondream', vram: '4GB', hint: 'Fast Vision' },
];

export const DEFAULT_VISION_MODEL = 'llava:7b';

export const DEFAULT_ANSWER_MODEL = 'llama3.1:8b-instruct-q4_0';

export interface QueryIntent {
  primary: 'temporal' | 'causal' | 'exploratory' | 'lookup';
  temporal?: TemporalIntent;
  causal?: boolean;
}

export interface TemporalIntent {
  range_type: string;
  start?: string;
  end?: string;
}

export interface SearchHistoryItem {
  id: string;
  query: string;
  timestamp: string;
  result_count: number;
}

// ============================================================================
// Connector Types
// ============================================================================

export interface Connector {
  id: string;
  name: string;
  connector_type: ConnectorType;
  status: ConnectorStatus;
  last_sync?: string;
  next_sync?: string;
  progress?: SyncProgress;
  error?: string;
  config: Record<string, unknown>;
  stats: ConnectorStats;
}

export type ConnectorType = 'local_folder' | 'obsidian' | 'github' | 'imap';
export type ConnectorStatus = 'active' | 'paused' | 'error' | 'syncing' | 'disabled';

export interface SyncProgress {
  current: number;
  total: number;
  phase: string;
}

export interface ConnectorStats {
  files_processed: number;
  entities_extracted: number;
  last_duration?: number;
}

export interface AddSourceRequest {
  connector_type: ConnectorType;
  name: string;
  config: Record<string, unknown>;
}

/** Result of a sync operation (e.g., GitHub clone/pull). */
export interface SyncResult {
  repo_id: string;
  full_name: string;
  status: 'completed' | 'failed' | 'in_progress' | 'pending';
  files_synced: number;
  bytes_synced: number;
  bytes_synced_mb: number;
  duration_seconds: number;
  branches_synced: string[];
  error_message?: string | null;
}

// ============================================================================
// Privacy Types
// ============================================================================

export interface ConsentRecord {
  source_id: string;
  source_name: string;
  consent_type: string; // Flexible to support Python backend scope-based types
  granted: boolean;
  granted_at?: string;
  expires_at?: string;
  retention_days?: number;
}

// Common consent types - the backend supports arbitrary scope strings
export type ConsentType = 'read' | 'process' | 'store' | 'cloud_backup' | string;

export interface GrantConsentRequest {
  source_id: string;
  consent_type: string; // Flexible to support Python backend scope-based types
  duration_days?: number;
  retention_days?: number;
}

export interface AuditLogEntry {
  id: string;
  timestamp: string;
  action: string;
  resource_type: string;
  resource_id?: string;
  details?: Record<string, unknown>;
  // Additional fields used by AuditLogViewer
  source?: string;
  job_id?: string;
  status?: string;
  chain_hash?: string;
  chain_prev?: string;
}

export interface AuditLogQuery {
  limit?: number;
  offset?: number;
  action_filter?: string;
  start_date?: string;
  end_date?: string;
}

// ============================================================================
// Cloud Sync Types
// ============================================================================

/**
 * Cloud sync consent scope values.
 * These match the Python CloudSyncScope enum.
 */
export type CloudSyncScope =
  | 'cloud:pkg:metadata_backup'
  | 'cloud:pkg:settings_backup'
  | 'cloud:search:history_sync';

/**
 * Cloud sync consent status from the backend.
 */
export interface CloudSyncConsentStatus {
  has_consent: boolean;
  granted_scopes: CloudSyncScope[];
  granted_at?: string;
  is_syncing: boolean;
  last_sync_at?: string;
}

/**
 * Request to grant cloud sync consent.
 */
export interface GrantCloudSyncRequest {
  scopes: CloudSyncScope[];
  operator?: string;
}

/**
 * Cloud sync audit log entry.
 */
export interface CloudSyncAuditEntry {
  id: string;
  timestamp: string;
  action: CloudSyncAuditAction;
  scope?: CloudSyncScope;
  nodes_affected: number;
  success: boolean;
  error_message?: string;
}

/**
 * Actions that can be recorded in cloud sync audit logs.
 */
export type CloudSyncAuditAction =
  | 'sync_started'
  | 'sync_completed'
  | 'sync_failed'
  | 'consent_granted'
  | 'consent_revoked'
  | 'data_deleted'
  | 'data_deletion_requested';

/**
 * Query parameters for cloud sync audit logs.
 */
export interface CloudSyncAuditQuery {
  limit?: number;
  action_filter?: CloudSyncAuditAction;
}

/**
 * Information about a cloud sync scope for UI display.
 */
export interface CloudSyncScopeInfo {
  scope: CloudSyncScope;
  title: string;
  description: string;
  required: boolean;
  default_enabled: boolean;
  data_shared?: string[];
  data_not_shared?: string[];
}

/**
 * Default scope information for the consent modal.
 */
export const CLOUD_SYNC_SCOPE_INFO: CloudSyncScopeInfo[] = [
  {
    scope: 'cloud:pkg:metadata_backup',
    title: 'Knowledge Graph Structure',
    description: 'Sync graph node labels, relationships, and timestamps (NOT document content)',
    required: true,
    default_enabled: true,
    data_shared: [
      'Entity names and types (Person, Organization, Concept, Event)',
      'Relationship types between entities',
      'Timestamps (created, modified)',
      'Source identifiers (which connector created the data)',
    ],
    data_not_shared: [
      'Document content',
      'Email bodies',
      'File contents',
      'Attachment data',
    ],
  },
  {
    scope: 'cloud:pkg:settings_backup',
    title: 'App Settings',
    description: 'Sync your Futurnal preferences and settings across devices',
    required: false,
    default_enabled: true,
    data_shared: [
      'Privacy level settings',
      'Connector configurations (without credentials)',
      'UI preferences',
      'Theme settings',
    ],
    data_not_shared: [
      'Passwords or API keys',
      'OAuth tokens',
      'Local file paths',
    ],
  },
  {
    scope: 'cloud:search:history_sync',
    title: 'Search History',
    description: 'Sync your search queries to continue research across devices',
    required: false,
    default_enabled: false,
    data_shared: [
      'Search query text',
      'Search timestamps',
      'Filter settings used',
    ],
    data_not_shared: [
      'Search results',
      'Document content from results',
    ],
  },
];

/**
 * Helper to get default enabled scopes.
 */
export function getDefaultEnabledScopes(): CloudSyncScope[] {
  return CLOUD_SYNC_SCOPE_INFO
    .filter(info => info.default_enabled)
    .map(info => info.scope);
}

/**
 * Helper to get required scopes.
 */
export function getRequiredScopes(): CloudSyncScope[] {
  return CLOUD_SYNC_SCOPE_INFO
    .filter(info => info.required)
    .map(info => info.scope);
}

// ============================================================================
// Orchestrator Types
// ============================================================================

/** Daemon status from `orchestrator daemon-status --json` */
export interface OrchestratorStatus {
  running: boolean;
  pid: number | null;
  workspace: string;
  stale_pid_file: boolean;
}

/** Full orchestrator status with job queue metrics (for future use) */
export interface OrchestratorFullStatus {
  running: boolean;
  active_jobs: number;
  pending_jobs: number;
  failed_jobs: number;
  sources: SourceStatus[];
  last_activity?: string;
  uptime_seconds: number;
}

export interface SourceStatus {
  id: string;
  name: string;
  status: string;
  progress?: SyncProgress;
  last_sync?: string;
  error?: string;
}

// ============================================================================
// Graph Types
// ============================================================================

export interface GraphNode {
  id: string;
  label: string;
  node_type: EntityType;
  timestamp?: string;
  metadata?: Record<string, unknown>;
  // Simulation coordinates (added by react-force-graph-2d during rendering)
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number;
  fy?: number;
}

export interface GraphLink {
  source: string;
  target: string;
  relationship: string;
  weight?: number;
  /** Confidence score for this relationship (0.0 - 1.0) */
  confidence?: number;
}

/** Pagination metadata for graph queries */
export interface PaginationMeta {
  offset: number;
  total: number;
  has_more: boolean;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
  pagination?: PaginationMeta;
}

/** Filter parameters for subgraph queries */
export interface GraphFilter {
  sources?: string[];
  node_types?: string[];
  min_confidence?: number;
  start_date?: string;
  end_date?: string;
}

/** Graph statistics */
export interface GraphStats {
  total_nodes: number;
  total_links: number;
  nodes_by_type: Record<string, number>;
  nodes_by_source: Record<string, number>;
}

// ============================================================================
// User Types
// ============================================================================

export interface User {
  id: string;
  email: string;
  displayName?: string;
  photoURL?: string;
  tier: UserTier;
  createdAt: string;
}

export type UserTier = 'free' | 'pro';

export interface UserTierLimits {
  maxSources: number;
  hasCloudBackup: boolean;
  // Phase 2 features
  hasEmergentInsights: boolean;
  hasCuriosityEngine: boolean;
  // Phase 3 features
  hasCausalExploration: boolean;
  hasAspirationalSelf: boolean;
}

export const TIER_LIMITS: Record<UserTier, UserTierLimits> = {
  free: {
    maxSources: 3,
    hasCloudBackup: false,
    hasEmergentInsights: false,
    hasCuriosityEngine: false,
    hasCausalExploration: false,
    hasAspirationalSelf: false,
  },
  pro: {
    maxSources: Infinity,
    hasCloudBackup: true,
    hasEmergentInsights: true,
    hasCuriosityEngine: true,
    hasCausalExploration: true,
    hasAspirationalSelf: true,
  },
};
