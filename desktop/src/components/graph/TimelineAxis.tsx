/**
 * TimelineAxis - SVG overlay for timeline layout axis and labels
 *
 * Renders time ticks at the top and entity type labels on the left.
 * Positioned as an overlay on top of the graph canvas.
 */

import type { EntityType } from '@/types/api';

interface TimelineAxisProps {
  /** Width of the container */
  width: number;
  /** Height of the container */
  height: number;
  /** Timeline tick positions and labels */
  ticks: { x: number; date: Date; label: string }[];
  /** Entity type Y positions */
  entityTypePositions: Record<EntityType, number>;
  /** X boundary for undated zone */
  undatedZoneBoundary: number;
  /** Padding used in layout */
  padding?: number;
}

/**
 * Entity type display labels
 */
const ENTITY_TYPE_LABELS: Record<EntityType, string> = {
  Source: 'Sources',
  Mailbox: 'Mailboxes',
  Document: 'Documents',
  Email: 'Emails',
  Code: 'Code',
  Person: 'People',
  Organization: 'Orgs',
  Event: 'Events',
  Concept: 'Concepts',
};

export function TimelineAxis({
  width,
  height,
  ticks,
  entityTypePositions,
  undatedZoneBoundary,
  padding = 60,
}: TimelineAxisProps) {
  // Only render if we have data
  if (Object.keys(entityTypePositions).length === 0) {
    return null;
  }

  return (
    <svg
      className="absolute inset-0 pointer-events-none"
      width={width}
      height={height}
      style={{ zIndex: 10 }}
    >
      <defs>
        <filter id="textShadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="black" floodOpacity="0.8" />
        </filter>
      </defs>

      {/* Undated zone background */}
      <rect
        x={0}
        y={0}
        width={undatedZoneBoundary}
        height={height}
        fill="rgba(255, 255, 255, 0.02)"
      />

      {/* Undated zone label */}
      <text
        x={padding + (undatedZoneBoundary - padding) / 2}
        y={padding / 2 + 5}
        textAnchor="middle"
        className="fill-white/40 text-[10px]"
        filter="url(#textShadow)"
      >
        Undated
      </text>

      {/* Undated zone separator */}
      <line
        x1={undatedZoneBoundary}
        y1={padding}
        x2={undatedZoneBoundary}
        y2={height - padding}
        stroke="rgba(255, 255, 255, 0.1)"
        strokeDasharray="4 4"
      />

      {/* Timeline axis line */}
      {ticks.length > 0 && (
        <line
          x1={ticks[0].x}
          y1={padding - 10}
          x2={ticks[ticks.length - 1].x}
          y2={padding - 10}
          stroke="rgba(255, 255, 255, 0.2)"
          strokeWidth={1}
        />
      )}

      {/* Timeline ticks */}
      {ticks.map((tick, index) => (
        <g key={index}>
          {/* Tick line */}
          <line
            x1={tick.x}
            y1={padding - 15}
            x2={tick.x}
            y2={padding - 5}
            stroke="rgba(255, 255, 255, 0.3)"
            strokeWidth={1}
          />
          {/* Tick label */}
          <text
            x={tick.x}
            y={padding - 20}
            textAnchor="middle"
            className="fill-white/50 text-[10px]"
            filter="url(#textShadow)"
          >
            {tick.label}
          </text>
          {/* Vertical guide line */}
          <line
            x1={tick.x}
            y1={padding}
            x2={tick.x}
            y2={height - padding}
            stroke="rgba(255, 255, 255, 0.05)"
            strokeWidth={1}
          />
        </g>
      ))}

      {/* Entity type labels and horizontal guides */}
      {(Object.entries(entityTypePositions) as [EntityType, number][]).map(
        ([type, y]) => (
          <g key={type}>
            {/* Horizontal guide line */}
            <line
              x1={padding}
              y1={y}
              x2={width - padding}
              y2={y}
              stroke="rgba(255, 255, 255, 0.03)"
              strokeWidth={1}
            />
            {/* Type label on left */}
            <text
              x={padding - 10}
              y={y}
              textAnchor="end"
              dominantBaseline="middle"
              className="fill-white/40 text-[10px]"
              filter="url(#textShadow)"
            >
              {ENTITY_TYPE_LABELS[type] || type}
            </text>
          </g>
        )
      )}

      {/* Timeline label */}
      <text
        x={width / 2}
        y={height - padding / 3}
        textAnchor="middle"
        className="fill-white/30 text-[11px] font-medium tracking-wide"
        filter="url(#textShadow)"
      >
        Timeline
      </text>
    </svg>
  );
}

export default TimelineAxis;
