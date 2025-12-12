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
}

export type EntityType = 'Event' | 'Document' | 'Code' | 'Person' | 'Concept' | 'Email' | 'Mailbox' | 'Source' | 'Organization';
export type SourceType = 'text' | 'ocr' | 'audio' | 'code';

export interface CausalChain {
  anchor: string;
  causes: string[];
  effects: string[];
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  query_id: string;
  intent: QueryIntent;
  execution_time_ms: number;
}

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
}

// Common consent types - the backend supports arbitrary scope strings
export type ConsentType = 'read' | 'process' | 'store' | 'cloud_backup' | string;

export interface GrantConsentRequest {
  source_id: string;
  consent_type: string; // Flexible to support Python backend scope-based types
  duration_days?: number;
}

export interface AuditLogEntry {
  id: string;
  timestamp: string;
  action: string;
  resource_type: string;
  resource_id?: string;
  details?: Record<string, unknown>;
}

export interface AuditLogQuery {
  limit?: number;
  offset?: number;
  action_filter?: string;
  start_date?: string;
  end_date?: string;
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
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
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
