/**
 * CausalChainPreview - Compact inline preview of causal relationships
 *
 * Displays causal context in format: ← causes → anchor → effects
 * Only shown when result has causal_chain data.
 */

import { GitBranch } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { CausalChain } from '@/types/api';

interface CausalChainPreviewProps {
  /** Causal chain data */
  chain: CausalChain;
  /** Additional class names */
  className?: string;
}

export function CausalChainPreview({
  chain,
  className,
}: CausalChainPreviewProps) {
  const { anchor, causes, effects } = chain;

  // Don't render if no causal data
  if (!anchor && causes.length === 0 && effects.length === 0) {
    return null;
  }

  return (
    <div
      data-slot="causal-chain-preview"
      className={cn(
        'p-2 rounded bg-white/5 border border-white/10',
        className
      )}
    >
      <div className="flex items-center gap-1.5 mb-1">
        <GitBranch className="h-3 w-3 text-white/40" />
        <span className="text-xs font-medium text-white/60">Causal Context</span>
      </div>

      <div className="text-xs text-white/70 flex items-center flex-wrap gap-1">
        {/* Causes (what led to this) */}
        {causes.length > 0 && (
          <>
            <span className="text-white/40">←</span>
            <span className="truncate max-w-[120px]" title={causes.join(', ')}>
              {causes.slice(0, 2).join(', ')}
              {causes.length > 2 && '...'}
            </span>
          </>
        )}

        {/* Anchor (the central event/entity) */}
        <span className="px-1.5 py-0.5 bg-white/10 rounded text-white/80 font-medium truncate max-w-[150px]">
          {anchor}
        </span>

        {/* Effects (what this leads to) */}
        {effects.length > 0 && (
          <>
            <span className="text-white/40">→</span>
            <span className="truncate max-w-[120px]" title={effects.join(', ')}>
              {effects.slice(0, 2).join(', ')}
              {effects.length > 2 && '...'}
            </span>
          </>
        )}
      </div>
    </div>
  );
}
