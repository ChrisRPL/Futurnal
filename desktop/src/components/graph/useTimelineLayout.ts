/**
 * useTimelineLayout - Hook for calculating timeline-based graph layout
 *
 * Positions nodes horizontally based on timestamp and vertically by entity type.
 * Supports time granularity (hour, day, week, month, year) for grouping nodes.
 * Nodes without timestamps are placed in an "Undated" zone on the left.
 */

import { useMemo } from 'react';
import type { GraphNode, EntityType } from '@/types/api';
import type { TimeGranularity } from '@/stores/graphStore';

interface TimelineLayoutOptions {
  /** Available width for the layout */
  width: number;
  /** Available height for the layout */
  height: number;
  /** Padding from edges */
  padding?: number;
  /** Width reserved for undated nodes */
  undatedZoneWidth?: number;
  /** Time granularity for grouping nodes */
  granularity?: TimeGranularity;
}

interface TimelineLayoutResult {
  /** Nodes with calculated positions */
  nodes: (GraphNode & { x: number; y: number })[];
  /** Time range of dated nodes */
  timeRange: { min: Date | null; max: Date | null };
  /** Entity type Y positions for axis labels (only types with nodes) */
  entityTypePositions: Record<EntityType, number>;
  /** Undated zone X boundary */
  undatedZoneBoundary: number;
  /** Tick positions for timeline axis */
  ticks: { x: number; date: Date; label: string }[];
  /** Number of undated nodes */
  undatedCount: number;
  /** Active entity types (types that have nodes) */
  activeEntityTypes: EntityType[];
}

/**
 * Entity type row order (top to bottom)
 * Grouped semantically: Sources → Content → Actors → Abstract
 */
const ENTITY_TYPE_ORDER: EntityType[] = [
  'Source',
  'Mailbox',
  'Document',
  'Email',
  'Code',
  'Person',
  'Organization',
  'Event',
  'Concept',
];

/**
 * Get the bucket key for a date based on granularity
 */
function getBucketKey(date: Date, granularity: TimeGranularity): string {
  switch (granularity) {
    case 'hour':
      return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}-${String(date.getHours()).padStart(2, '0')}`;
    case 'day':
      return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
    case 'week': {
      // Get the Monday of the week
      const d = new Date(date);
      const day = d.getDay();
      const diff = d.getDate() - day + (day === 0 ? -6 : 1);
      d.setDate(diff);
      return `${d.getFullYear()}-W${String(Math.ceil((d.getDate() + 6) / 7)).padStart(2, '0')}`;
    }
    case 'month':
      return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
    case 'year':
      return `${date.getFullYear()}`;
    case 'auto':
    default:
      // For auto, use day as default bucket
      return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
  }
}

/**
 * Get the start date of a bucket
 */
function getBucketStartDate(date: Date, granularity: TimeGranularity): Date {
  const d = new Date(date);
  switch (granularity) {
    case 'hour':
      d.setMinutes(0, 0, 0);
      return d;
    case 'day':
      d.setHours(0, 0, 0, 0);
      return d;
    case 'week': {
      const day = d.getDay();
      const diff = d.getDate() - day + (day === 0 ? -6 : 1);
      d.setDate(diff);
      d.setHours(0, 0, 0, 0);
      return d;
    }
    case 'month':
      d.setDate(1);
      d.setHours(0, 0, 0, 0);
      return d;
    case 'year':
      d.setMonth(0, 1);
      d.setHours(0, 0, 0, 0);
      return d;
    case 'auto':
    default:
      d.setHours(0, 0, 0, 0);
      return d;
  }
}

/**
 * Format a tick label based on granularity
 */
function formatTickLabel(date: Date, granularity: TimeGranularity): string {
  switch (granularity) {
    case 'hour':
      return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    case 'day':
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    case 'week':
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    case 'month':
      return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short' });
    case 'year':
      return date.getFullYear().toString();
    case 'auto':
    default:
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  }
}

/**
 * Determine best granularity based on time range
 */
function determineAutoGranularity(minDate: Date, maxDate: Date): TimeGranularity {
  const diffMs = maxDate.getTime() - minDate.getTime();
  const diffDays = diffMs / (1000 * 60 * 60 * 24);

  if (diffDays < 2) return 'hour';
  if (diffDays < 60) return 'day';
  if (diffDays < 180) return 'week';
  if (diffDays < 730) return 'month';
  return 'year';
}

/**
 * Calculate timeline layout positions for nodes
 */
export function useTimelineLayout(
  nodes: GraphNode[],
  options: TimelineLayoutOptions
): TimelineLayoutResult {
  const {
    width,
    height,
    padding = 60,
    undatedZoneWidth = 100,
    granularity = 'auto',
  } = options;

  return useMemo(() => {
    if (!nodes.length) {
      return {
        nodes: [],
        timeRange: { min: null, max: null },
        entityTypePositions: {} as Record<EntityType, number>,
        undatedZoneBoundary: undatedZoneWidth + padding,
        ticks: [],
        undatedCount: 0,
        activeEntityTypes: [],
      };
    }

    // Separate dated and undated nodes
    const datedNodes: (GraphNode & { date: Date })[] = [];
    const undatedNodes: GraphNode[] = [];

    nodes.forEach((node) => {
      if (node.timestamp) {
        const date = new Date(node.timestamp);
        if (!isNaN(date.getTime())) {
          datedNodes.push({ ...node, date });
        } else {
          undatedNodes.push(node);
        }
      } else {
        undatedNodes.push(node);
      }
    });

    // Calculate time range
    let minDate: Date | null = null;
    let maxDate: Date | null = null;

    if (datedNodes.length > 0) {
      const dates = datedNodes.map((n) => n.date.getTime());
      minDate = new Date(Math.min(...dates));
      maxDate = new Date(Math.max(...dates));
    }

    // Determine effective granularity
    const effectiveGranularity = granularity === 'auto' && minDate && maxDate
      ? determineAutoGranularity(minDate, maxDate)
      : granularity === 'auto' ? 'day' : granularity;

    // Extend time range to bucket boundaries
    if (minDate && maxDate) {
      minDate = getBucketStartDate(minDate, effectiveGranularity);
      // Add one more bucket after max for padding
      const tempMax = new Date(maxDate);
      switch (effectiveGranularity) {
        case 'hour':
          tempMax.setHours(tempMax.getHours() + 1);
          break;
        case 'day':
          tempMax.setDate(tempMax.getDate() + 1);
          break;
        case 'week':
          tempMax.setDate(tempMax.getDate() + 7);
          break;
        case 'month':
          tempMax.setMonth(tempMax.getMonth() + 1);
          break;
        case 'year':
          tempMax.setFullYear(tempMax.getFullYear() + 1);
          break;
      }
      maxDate = getBucketStartDate(tempMax, effectiveGranularity);
    }

    // Find active entity types (types that have nodes)
    const activeTypeSet = new Set<EntityType>();
    nodes.forEach((node) => {
      const type = node.node_type || 'Document';
      activeTypeSet.add(type);
    });
    const activeEntityTypes = ENTITY_TYPE_ORDER.filter(t => activeTypeSet.has(t));

    // Calculate Y positions for active entity types only
    // Guard against negative usable dimensions
    const usableHeight = Math.max(100, height - padding * 2);
    const rowCount = Math.max(activeEntityTypes.length, 1);
    const rowHeight = usableHeight / rowCount;
    const entityTypePositions: Record<EntityType, number> = {} as Record<EntityType, number>;

    activeEntityTypes.forEach((type, index) => {
      entityTypePositions[type] = padding + rowHeight * (index + 0.5);
    });

    // Calculate X positions - guard against negative timelineWidth
    const undatedZoneBoundary = undatedZoneWidth + padding;
    const timelineStart = undatedZoneBoundary + padding;
    const timelineEnd = Math.max(timelineStart + 100, width - padding);
    const timelineWidth = Math.max(100, timelineEnd - timelineStart);

    // Position nodes
    const positionedNodes: (GraphNode & { x: number; y: number })[] = [];

    // Position undated nodes in a vertical stack by type
    const undatedByType: Map<EntityType, GraphNode[]> = new Map();
    undatedNodes.forEach((node) => {
      const type = node.node_type || 'Document';
      if (!undatedByType.has(type)) {
        undatedByType.set(type, []);
      }
      undatedByType.get(type)!.push(node);
    });

    undatedByType.forEach((nodesOfType, type) => {
      const baseY = entityTypePositions[type] || height / 2;
      const spacing = Math.min(20, (rowHeight * 0.8) / (nodesOfType.length + 1));

      nodesOfType.forEach((node, index) => {
        // Distribute within the undated zone - deterministic position based on index
        const xOffset = ((index % 3) - 1) * (undatedZoneWidth * 0.25);
        const yOffset = (index - (nodesOfType.length - 1) / 2) * spacing;

        positionedNodes.push({
          ...node,
          x: padding + undatedZoneWidth / 2 + xOffset,
          y: baseY + yOffset,
        });
      });
    });

    // Position dated nodes along timeline grouped by bucket
    if (minDate && maxDate) {
      const timeRange = maxDate.getTime() - minDate.getTime();

      // Group by bucket key AND entity type
      const buckets: Map<string, (GraphNode & { date: Date })[]> = new Map();

      datedNodes.forEach((node) => {
        const type = node.node_type || 'Document';
        const bucketKey = getBucketKey(node.date, effectiveGranularity);
        const key = `${type}-${bucketKey}`;
        if (!buckets.has(key)) {
          buckets.set(key, []);
        }
        buckets.get(key)!.push(node);
      });

      buckets.forEach((nodesInBucket) => {
        const type = nodesInBucket[0].node_type || 'Document';
        const baseY = entityTypePositions[type] || height / 2;

        // Calculate X position based on bucket center
        const bucketStart = getBucketStartDate(nodesInBucket[0].date, effectiveGranularity);
        const timeOffset = bucketStart.getTime() - minDate!.getTime();
        const xRatio = timeRange > 0 ? timeOffset / timeRange : 0.5;
        const bucketX = timelineStart + xRatio * timelineWidth;

        // Stack nodes within the bucket
        const nodeCount = nodesInBucket.length;
        const spacing = Math.min(12, (rowHeight * 0.6) / (nodeCount + 1));

        nodesInBucket.forEach((node, index) => {
          // Stack vertically within the row
          const yOffset = (index - (nodeCount - 1) / 2) * spacing;
          // Small X offset for nodes in same bucket to avoid perfect overlap
          const xJitter = nodeCount > 1 ? ((index % 3) - 1) * 8 : 0;

          positionedNodes.push({
            ...node,
            x: bucketX + xJitter,
            y: baseY + yOffset,
          });
        });
      });
    }

    // Generate timeline ticks based on granularity
    const ticks: { x: number; date: Date; label: string }[] = [];

    if (minDate && maxDate) {
      const timeRange = maxDate.getTime() - minDate.getTime();

      // Calculate tick interval based on granularity
      let tickInterval: number;
      switch (effectiveGranularity) {
        case 'hour':
          tickInterval = 1000 * 60 * 60; // 1 hour
          break;
        case 'day':
          tickInterval = 1000 * 60 * 60 * 24; // 1 day
          break;
        case 'week':
          tickInterval = 1000 * 60 * 60 * 24 * 7; // 1 week
          break;
        case 'month':
          tickInterval = 1000 * 60 * 60 * 24 * 30; // ~1 month
          break;
        case 'year':
          tickInterval = 1000 * 60 * 60 * 24 * 365; // ~1 year
          break;
        default:
          tickInterval = 1000 * 60 * 60 * 24; // 1 day
      }

      // Limit number of ticks
      const maxTicks = Math.min(12, Math.max(3, Math.floor(timelineWidth / 80)));
      const tickCount = Math.min(maxTicks, Math.ceil(timeRange / tickInterval));
      const adjustedInterval = timeRange / tickCount;

      for (let i = 0; i <= tickCount; i++) {
        const tickTime = minDate.getTime() + i * adjustedInterval;
        const date = new Date(tickTime);
        const x = timelineStart + (i / tickCount) * timelineWidth;

        ticks.push({
          x,
          date,
          label: formatTickLabel(date, effectiveGranularity),
        });
      }
    }

    return {
      nodes: positionedNodes,
      timeRange: { min: minDate, max: maxDate },
      entityTypePositions,
      undatedZoneBoundary,
      ticks,
      undatedCount: undatedNodes.length,
      activeEntityTypes,
    };
  }, [nodes, width, height, padding, undatedZoneWidth, granularity]);
}

export default useTimelineLayout;
