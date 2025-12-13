/**
 * Cloud Sync Client - Firebase Firestore sync for PKG metadata
 *
 * Handles synchronization of Personal Knowledge Graph metadata to Firebase.
 * Only syncs metadata (node labels, types, relationships) - NEVER document content.
 *
 * Privacy-first design:
 * - Document content never leaves the device
 * - Only graph structure (labels, types, timestamps) is synced
 * - All sync operations require explicit consent
 * - Revocation triggers immediate cloud data deletion
 */

import {
  collection,
  doc,
  getDocs,
  writeBatch,
  deleteDoc,
  setDoc,
  serverTimestamp,
  query,
  orderBy,
  limit,
  Timestamp,
  type DocumentData,
} from 'firebase/firestore';
import { db, auth } from './firebase';
import { graphApi } from './api';
import type { GraphNode, GraphLink, CloudSyncScope } from '@/types/api';

// ============================================================================
// Firestore Schema Types
// ============================================================================

/**
 * PKG node document stored in Firestore.
 * Contains ONLY metadata - no document content.
 */
interface PKGSyncNode {
  nodeId: string;
  nodeType: string;
  label: string;
  createdAt: Timestamp | null;
  updatedAt: Timestamp;
  sourceType: string | null;
  syncVersion: number;
  lastSyncedAt: Timestamp;
}

/**
 * PKG relationship document stored in Firestore.
 */
interface PKGSyncRelationship {
  source: string;
  target: string;
  relationship: string;
  confidence: number | null;
  syncVersion: number;
  lastSyncedAt: Timestamp;
}

/**
 * Sync status document for tracking sync state.
 */
interface SyncStatusDoc {
  lastSyncAt: Timestamp;
  nodeCount: number;
  relationshipCount: number;
  syncVersion: number;
  deviceId: string;
}

/**
 * Result of a sync operation.
 */
export interface CloudSyncResult {
  success: boolean;
  nodesSynced: number;
  relationshipsSynced: number;
  nodesDeleted: number;
  durationMs: number;
  error?: string;
}

// ============================================================================
// Constants
// ============================================================================

/** Current sync version - increment when schema changes */
const SYNC_VERSION = 1;

/** Maximum nodes per batch write (Firestore limit is 500) */
const BATCH_SIZE = 400;

/** Device identifier for multi-device tracking */
const DEVICE_ID = `desktop-${Date.now()}`;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get the current authenticated user ID.
 * @throws Error if user is not authenticated
 */
function getCurrentUserId(): string {
  const user = auth.currentUser;
  if (!user) {
    throw new Error('User not authenticated - cannot perform cloud sync');
  }
  return user.uid;
}

/**
 * Get Firestore collection reference for user's PKG nodes.
 */
function getNodesCollection(userId: string) {
  return collection(db, `users/${userId}/pkg_nodes`);
}

/**
 * Get Firestore collection reference for user's PKG relationships.
 */
function getRelationshipsCollection(userId: string) {
  return collection(db, `users/${userId}/pkg_relationships`);
}

/**
 * Get Firestore document reference for user's sync status.
 */
function getSyncStatusDoc(userId: string) {
  return doc(db, `users/${userId}/sync_status/current`);
}

/**
 * Convert a GraphNode to Firestore sync format.
 * Strips any content-like fields and keeps only metadata.
 */
function nodeToSyncFormat(node: GraphNode): PKGSyncNode {
  return {
    nodeId: node.id,
    nodeType: node.node_type,
    label: node.label,
    createdAt: node.timestamp ? Timestamp.fromDate(new Date(node.timestamp)) : null,
    updatedAt: serverTimestamp() as Timestamp,
    sourceType: node.metadata?.source_type as string | null ?? null,
    syncVersion: SYNC_VERSION,
    lastSyncedAt: serverTimestamp() as Timestamp,
  };
}

/**
 * Convert a GraphLink to Firestore sync format.
 */
function linkToSyncFormat(link: GraphLink): PKGSyncRelationship {
  return {
    source: typeof link.source === 'string' ? link.source : (link.source as unknown as { id: string }).id,
    target: typeof link.target === 'string' ? link.target : (link.target as unknown as { id: string }).id,
    relationship: link.relationship,
    confidence: link.confidence ?? null,
    syncVersion: SYNC_VERSION,
    lastSyncedAt: serverTimestamp() as Timestamp,
  };
}

/**
 * Generate a stable ID for a relationship (for deduplication).
 */
function getRelationshipId(link: GraphLink): string {
  const source = typeof link.source === 'string' ? link.source : (link.source as unknown as { id: string }).id;
  const target = typeof link.target === 'string' ? link.target : (link.target as unknown as { id: string }).id;
  return `${source}__${link.relationship}__${target}`;
}

// ============================================================================
// Main Sync Functions
// ============================================================================

/**
 * Perform a full cloud sync of PKG metadata to Firestore.
 *
 * This function:
 * 1. Fetches the local knowledge graph
 * 2. Uploads all node metadata to Firestore (in batches)
 * 3. Uploads all relationships to Firestore (in batches)
 * 4. Updates sync status document
 *
 * @param scopes - Granted consent scopes (currently only PKG_METADATA_BACKUP supported)
 * @returns Sync result with statistics
 */
export async function performCloudSync(scopes: CloudSyncScope[]): Promise<CloudSyncResult> {
  const startTime = Date.now();

  // Verify consent for metadata backup
  if (!scopes.includes('cloud:pkg:metadata_backup')) {
    return {
      success: false,
      nodesSynced: 0,
      relationshipsSynced: 0,
      nodesDeleted: 0,
      durationMs: Date.now() - startTime,
      error: 'No consent for metadata backup',
    };
  }

  try {
    const userId = getCurrentUserId();
    console.log('[CloudSync] Starting sync for user:', userId);

    // Fetch local graph data
    const graphData = await graphApi.getGraph(5000);
    const nodes = graphData.nodes;
    const links = graphData.links;

    console.log(`[CloudSync] Syncing ${nodes.length} nodes and ${links.length} relationships`);

    // Get existing cloud node IDs for deletion detection
    const existingNodeIds = new Set<string>();
    const existingRelIds = new Set<string>();

    const nodesSnapshot = await getDocs(getNodesCollection(userId));
    nodesSnapshot.forEach((doc) => existingNodeIds.add(doc.id));

    const relsSnapshot = await getDocs(getRelationshipsCollection(userId));
    relsSnapshot.forEach((doc) => existingRelIds.add(doc.id));

    // Calculate which nodes to delete (in cloud but not in local)
    const localNodeIds = new Set(nodes.map((n) => n.id));
    const nodesToDelete = [...existingNodeIds].filter((id) => !localNodeIds.has(id));

    const localRelIds = new Set(links.map((l) => getRelationshipId(l)));
    const relsToDelete = [...existingRelIds].filter((id) => !localRelIds.has(id));

    let nodesSynced = 0;
    let relationshipsSynced = 0;
    let nodesDeleted = 0;

    // Batch write nodes
    for (let i = 0; i < nodes.length; i += BATCH_SIZE) {
      const batch = writeBatch(db);
      const batchNodes = nodes.slice(i, i + BATCH_SIZE);

      for (const node of batchNodes) {
        const docRef = doc(getNodesCollection(userId), node.id);
        batch.set(docRef, nodeToSyncFormat(node));
        nodesSynced++;
      }

      await batch.commit();
      console.log(`[CloudSync] Committed node batch ${Math.ceil((i + 1) / BATCH_SIZE)}`);
    }

    // Batch write relationships
    for (let i = 0; i < links.length; i += BATCH_SIZE) {
      const batch = writeBatch(db);
      const batchLinks = links.slice(i, i + BATCH_SIZE);

      for (const link of batchLinks) {
        const relId = getRelationshipId(link);
        const docRef = doc(getRelationshipsCollection(userId), relId);
        batch.set(docRef, linkToSyncFormat(link));
        relationshipsSynced++;
      }

      await batch.commit();
      console.log(`[CloudSync] Committed relationship batch ${Math.ceil((i + 1) / BATCH_SIZE)}`);
    }

    // Delete orphaned nodes and relationships
    for (let i = 0; i < nodesToDelete.length; i += BATCH_SIZE) {
      const batch = writeBatch(db);
      const batchDelete = nodesToDelete.slice(i, i + BATCH_SIZE);

      for (const nodeId of batchDelete) {
        batch.delete(doc(getNodesCollection(userId), nodeId));
        nodesDeleted++;
      }

      await batch.commit();
    }

    for (let i = 0; i < relsToDelete.length; i += BATCH_SIZE) {
      const batch = writeBatch(db);
      const batchDelete = relsToDelete.slice(i, i + BATCH_SIZE);

      for (const relId of batchDelete) {
        batch.delete(doc(getRelationshipsCollection(userId), relId));
      }

      await batch.commit();
    }

    // Update sync status
    const statusDoc: SyncStatusDoc = {
      lastSyncAt: serverTimestamp() as Timestamp,
      nodeCount: nodesSynced,
      relationshipCount: relationshipsSynced,
      syncVersion: SYNC_VERSION,
      deviceId: DEVICE_ID,
    };

    await setDoc(getSyncStatusDoc(userId), statusDoc);

    const durationMs = Date.now() - startTime;
    console.log(`[CloudSync] Sync completed in ${durationMs}ms`);

    return {
      success: true,
      nodesSynced,
      relationshipsSynced,
      nodesDeleted,
      durationMs,
    };
  } catch (error) {
    const durationMs = Date.now() - startTime;
    const errorMessage = error instanceof Error ? error.message : 'Unknown sync error';
    console.error('[CloudSync] Sync failed:', error);

    return {
      success: false,
      nodesSynced: 0,
      relationshipsSynced: 0,
      nodesDeleted: 0,
      durationMs,
      error: errorMessage,
    };
  }
}

/**
 * Delete ALL cloud data for the current user.
 * Called when consent is revoked - this is a privacy requirement.
 *
 * @returns Number of documents deleted
 */
export async function deleteAllCloudData(): Promise<number> {
  try {
    const userId = getCurrentUserId();
    console.log('[CloudSync] Deleting all cloud data for user:', userId);

    let totalDeleted = 0;

    // Delete all nodes
    const nodesSnapshot = await getDocs(getNodesCollection(userId));
    for (let i = 0; i < nodesSnapshot.docs.length; i += BATCH_SIZE) {
      const batch = writeBatch(db);
      const batchDocs = nodesSnapshot.docs.slice(i, i + BATCH_SIZE);

      for (const docSnap of batchDocs) {
        batch.delete(docSnap.ref);
        totalDeleted++;
      }

      await batch.commit();
    }

    // Delete all relationships
    const relsSnapshot = await getDocs(getRelationshipsCollection(userId));
    for (let i = 0; i < relsSnapshot.docs.length; i += BATCH_SIZE) {
      const batch = writeBatch(db);
      const batchDocs = relsSnapshot.docs.slice(i, i + BATCH_SIZE);

      for (const docSnap of batchDocs) {
        batch.delete(docSnap.ref);
        totalDeleted++;
      }

      await batch.commit();
    }

    // Delete sync status
    await deleteDoc(getSyncStatusDoc(userId));

    console.log(`[CloudSync] Deleted ${totalDeleted} documents from cloud`);
    return totalDeleted;
  } catch (error) {
    console.error('[CloudSync] Failed to delete cloud data:', error);
    throw error;
  }
}

/**
 * Get the current sync status from Firestore.
 * Returns null if no sync has been performed yet.
 */
export async function getCloudSyncStatus(): Promise<SyncStatusDoc | null> {
  try {
    const userId = getCurrentUserId();
    const statusRef = getSyncStatusDoc(userId);
    const statusSnap = await getDocs(query(collection(statusRef.parent, statusRef.id)));

    // Actually get the single document
    const docRef = doc(db, `users/${userId}/sync_status/current`);
    const { getDoc } = await import('firebase/firestore');
    const docSnap = await getDoc(docRef);

    if (docSnap.exists()) {
      return docSnap.data() as SyncStatusDoc;
    }
    return null;
  } catch (error) {
    console.error('[CloudSync] Failed to get sync status:', error);
    return null;
  }
}

/**
 * Get the count of nodes and relationships in cloud storage.
 * Useful for displaying sync status in UI.
 */
export async function getCloudDataCounts(): Promise<{ nodes: number; relationships: number }> {
  try {
    const userId = getCurrentUserId();

    const nodesSnapshot = await getDocs(
      query(getNodesCollection(userId), limit(1))
    );
    const relsSnapshot = await getDocs(
      query(getRelationshipsCollection(userId), limit(1))
    );

    // Get actual counts from sync status if available
    const status = await getCloudSyncStatus();
    if (status) {
      return {
        nodes: status.nodeCount,
        relationships: status.relationshipCount,
      };
    }

    // Fall back to snapshot counts (may not reflect total)
    return {
      nodes: nodesSnapshot.size,
      relationships: relsSnapshot.size,
    };
  } catch (error) {
    console.error('[CloudSync] Failed to get cloud data counts:', error);
    return { nodes: 0, relationships: 0 };
  }
}

/**
 * Check if user has any data in cloud storage.
 */
export async function hasCloudData(): Promise<boolean> {
  try {
    const userId = getCurrentUserId();
    const nodesSnapshot = await getDocs(
      query(getNodesCollection(userId), limit(1))
    );
    return !nodesSnapshot.empty;
  } catch {
    return false;
  }
}
