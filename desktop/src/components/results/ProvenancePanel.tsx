/**
 * ProvenancePanel - Collapsible metadata panel for search results
 *
 * Displays provenance information:
 * - Source file/location
 * - Extraction timestamp
 * - Schema version
 * - Entity ID
 */

import { FileText, Clock, Database, Hash } from 'lucide-react';
import { cn } from '@/lib/utils';
import { formatDateTime } from '@/lib/utils';

interface ProvenancePanelProps {
  /** Result metadata containing provenance information */
  metadata: Record<string, unknown>;
  /** Additional class names */
  className?: string;
}

interface ProvenanceItem {
  icon: typeof FileText;
  label: string;
  value: string | undefined;
}

/**
 * Truncate an ID for display, showing first 6 and last 4 characters.
 */
function truncateId(id: string | undefined): string {
  if (!id) return '';
  if (id.length <= 12) return id;
  return `${id.slice(0, 6)}...${id.slice(-4)}`;
}

export function ProvenancePanel({ metadata, className }: ProvenancePanelProps) {
  // Build provenance items from metadata
  const items: ProvenanceItem[] = [
    {
      icon: FileText,
      label: 'Source',
      value: metadata.source as string | undefined,
    },
    {
      icon: Clock,
      label: 'Extracted',
      value: metadata.extractionTimestamp
        ? formatDateTime(metadata.extractionTimestamp as string)
        : undefined,
    },
    {
      icon: Database,
      label: 'Schema',
      value: metadata.schemaVersion
        ? `v${metadata.schemaVersion}`
        : undefined,
    },
    {
      icon: Hash,
      label: 'Entity ID',
      value: truncateId(metadata.entityId as string | undefined),
    },
  ].filter((item) => item.value);

  if (items.length === 0) {
    return null;
  }

  return (
    <div
      data-slot="provenance-panel"
      className={cn(
        'p-3 rounded bg-white/[0.03] border border-white/10',
        className
      )}
    >
      <div className="text-xs font-medium text-white/50 mb-2">Provenance</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-2">
        {items.map((item) => (
          <div key={item.label} className="flex items-center gap-2 text-xs min-w-0">
            <item.icon className="h-3 w-3 text-white/30 flex-shrink-0" />
            <span className="text-white/40 flex-shrink-0">{item.label}:</span>
            <span
              className="text-white/60 truncate"
              title={item.value}
            >
              {item.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
