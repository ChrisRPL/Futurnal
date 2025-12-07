Summary: Implement search results display with provenance metadata, entity badges, and quick actions.

# 06 · Results & Provenance View

## Purpose

Create the search results display component showing result cards with snippets, relevance scores, entity type badges, source type indicators, timestamps, confidence scores, and provenance metadata. Include quick actions for opening sources, copying, saving, and sharing results.

**Criticality**: HIGH - Core search experience visualization

## Scope

- Result card component with snippet highlighting
- Relevance score indicator
- Entity type badges (Event, Document, Code, Person)
- Source type indicators (text/OCR/audio)
- Timestamp display
- Confidence score visualization
- Expandable context view
- Provenance metadata panel
- Quick actions: Open source, Copy, Save, Share
- Causal chain visualization (for causal queries)
- Multimodal source badges with quality indicators

## Requirements Alignment

- **Feature Requirement**: "Results view displays snippets, provenance metadata, and quick actions"
- **Search API Integration**: Maps to SearchResult from Hybrid Search API

## Component Design

### Result Card Component

```tsx
// src/components/results/ResultCard.tsx
import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Calendar,
  FileText,
  User,
  Code,
  Mic,
  Image,
  ExternalLink,
  Copy,
  Bookmark,
  Share2,
  ChevronDown,
  ChevronUp,
  Sparkles,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { ProvenancePanel } from './ProvenancePanel';

interface SearchResult {
  id: string;
  content: string;
  score: float;
  confidence: number;
  timestamp?: string;
  entityType?: 'Event' | 'Document' | 'Code' | 'Person';
  sourceType?: 'text' | 'ocr' | 'audio' | 'code';
  sourceConfidence?: number;
  causalChain?: {
    anchor: string;
    causes: string[];
    effects: string[];
  };
  metadata: {
    source?: string;
    extractionTimestamp?: string;
    schemaVersion?: number;
    [key: string]: unknown;
  };
}

interface ResultCardProps {
  result: SearchResult;
  query: string;
  onSelect?: () => void;
}

const ENTITY_ICONS: Record<string, typeof FileText> = {
  Event: Calendar,
  Document: FileText,
  Code: Code,
  Person: User,
};

const SOURCE_ICONS: Record<string, typeof FileText> = {
  text: FileText,
  ocr: Image,
  audio: Mic,
  code: Code,
};

const ENTITY_COLORS: Record<string, string> = {
  Event: 'default',
  Document: 'secondary',
  Code: 'accent',
  Person: 'success',
};

export function ResultCard({ result, query, onSelect }: ResultCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showProvenance, setShowProvenance] = useState(false);

  const EntityIcon = result.entityType ? ENTITY_ICONS[result.entityType] : FileText;
  const SourceIcon = result.sourceType ? SOURCE_ICONS[result.sourceType] : FileText;

  // Highlight query terms in content
  const highlightedContent = highlightTerms(result.content, query);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(result.content);
    // Show toast notification
  };

  const handleOpenSource = () => {
    // Open source file/location via Tauri
  };

  return (
    <Card
      className={cn(
        'group transition-all duration-150',
        'hover:border-border-hover hover:bg-background-elevated/50',
        'cursor-pointer'
      )}
      onClick={onSelect}
    >
      <CardContent className="p-4">
        {/* Header Row */}
        <div className="flex items-start justify-between gap-3 mb-2">
          <div className="flex items-center gap-2 flex-wrap">
            {/* Entity Type Badge */}
            {result.entityType && (
              <Badge variant={ENTITY_COLORS[result.entityType] as any} className="gap-1">
                <EntityIcon className="h-3 w-3" />
                {result.entityType}
              </Badge>
            )}

            {/* Source Type Badge */}
            {result.sourceType && result.sourceType !== 'text' && (
              <Badge variant="outline" className="gap-1">
                <SourceIcon className="h-3 w-3" />
                {result.sourceType.toUpperCase()}
                {result.sourceConfidence && (
                  <span className="opacity-60">
                    {Math.round(result.sourceConfidence * 100)}%
                  </span>
                )}
              </Badge>
            )}

            {/* Timestamp */}
            {result.timestamp && (
              <span className="text-xs text-text-tertiary flex items-center gap-1">
                <Calendar className="h-3 w-3" />
                {formatTimestamp(result.timestamp)}
              </span>
            )}
          </div>

          {/* Score Indicator */}
          <div className="flex items-center gap-2">
            <ScoreIndicator score={result.score} />
            <ConfidenceIndicator confidence={result.confidence} />
          </div>
        </div>

        {/* Content Snippet */}
        <div
          className={cn(
            'text-sm text-text-primary leading-relaxed',
            !isExpanded && 'line-clamp-3'
          )}
          dangerouslySetInnerHTML={{ __html: highlightedContent }}
        />

        {/* Expand/Collapse for long content */}
        {result.content.length > 200 && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setIsExpanded(!isExpanded);
            }}
            className="mt-2 text-xs text-primary hover:underline flex items-center gap-1"
          >
            {isExpanded ? (
              <>
                <ChevronUp className="h-3 w-3" /> Show less
              </>
            ) : (
              <>
                <ChevronDown className="h-3 w-3" /> Show more
              </>
            )}
          </button>
        )}

        {/* Causal Chain (if present) */}
        {result.causalChain && (
          <CausalChainPreview chain={result.causalChain} />
        )}

        {/* Quick Actions */}
        <div className="flex items-center justify-between mt-3 pt-3 border-t border-border">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowProvenance(!showProvenance);
            }}
            className="text-xs text-text-tertiary hover:text-text-secondary flex items-center gap-1"
          >
            <Sparkles className="h-3 w-3" />
            {showProvenance ? 'Hide' : 'Show'} provenance
          </button>

          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={(e) => {
                e.stopPropagation();
                handleOpenSource();
              }}
              title="Open source"
            >
              <ExternalLink className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={(e) => {
                e.stopPropagation();
                handleCopy();
              }}
              title="Copy"
            >
              <Copy className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={(e) => e.stopPropagation()}
              title="Save"
            >
              <Bookmark className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={(e) => e.stopPropagation()}
              title="Share"
            >
              <Share2 className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>

        {/* Provenance Panel */}
        {showProvenance && (
          <ProvenancePanel metadata={result.metadata} className="mt-3" />
        )}
      </CardContent>
    </Card>
  );
}

// Score Indicator Component
function ScoreIndicator({ score }: { score: number }) {
  const percentage = Math.round(score * 100);
  const color =
    percentage >= 80
      ? 'text-secondary'
      : percentage >= 60
      ? 'text-primary'
      : 'text-text-tertiary';

  return (
    <div className={cn('text-xs font-medium', color)} title="Relevance score">
      {percentage}%
    </div>
  );
}

// Confidence Indicator Component
function ConfidenceIndicator({ confidence }: { confidence: number }) {
  const bars = 3;
  const filledBars = Math.ceil(confidence * bars);

  return (
    <div className="flex gap-0.5" title={`Confidence: ${Math.round(confidence * 100)}%`}>
      {Array.from({ length: bars }).map((_, i) => (
        <div
          key={i}
          className={cn(
            'w-1 h-3 rounded-sm',
            i < filledBars ? 'bg-primary' : 'bg-border'
          )}
        />
      ))}
    </div>
  );
}

// Causal Chain Preview
function CausalChainPreview({ chain }: { chain: SearchResult['causalChain'] }) {
  if (!chain) return null;

  return (
    <div className="mt-3 p-2 rounded-md bg-accent/10 border border-accent/20">
      <div className="text-xs font-medium text-accent mb-1">Causal Context</div>
      <div className="text-xs text-text-secondary">
        {chain.causes.length > 0 && (
          <span>← {chain.causes.slice(0, 2).join(', ')}</span>
        )}
        <span className="mx-2 text-accent font-medium">{chain.anchor}</span>
        {chain.effects.length > 0 && (
          <span>→ {chain.effects.slice(0, 2).join(', ')}</span>
        )}
      </div>
    </div>
  );
}

// Utility Functions
function highlightTerms(content: string, query: string): string {
  if (!query) return content;
  const terms = query.split(/\s+/).filter(Boolean);
  let result = content;
  terms.forEach((term) => {
    const regex = new RegExp(`(${term})`, 'gi');
    result = result.replace(regex, '<mark class="bg-primary/30 rounded px-0.5">$1</mark>');
  });
  return result;
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return 'Today';
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  return date.toLocaleDateString();
}
```

### Provenance Panel Component

```tsx
// src/components/results/ProvenancePanel.tsx
import { cn } from '@/lib/utils';
import { FileText, Clock, Database, Hash } from 'lucide-react';

interface ProvenancePanelProps {
  metadata: Record<string, unknown>;
  className?: string;
}

export function ProvenancePanel({ metadata, className }: ProvenancePanelProps) {
  const items = [
    { icon: FileText, label: 'Source', value: metadata.source as string },
    { icon: Clock, label: 'Extracted', value: formatDate(metadata.extractionTimestamp as string) },
    { icon: Database, label: 'Schema', value: `v${metadata.schemaVersion}` },
    { icon: Hash, label: 'Entity ID', value: truncateId(metadata.entityId as string) },
  ].filter((item) => item.value);

  return (
    <div
      className={cn(
        'p-3 rounded-md bg-background-deep border border-border',
        className
      )}
    >
      <div className="text-xs font-medium text-text-secondary mb-2">Provenance</div>
      <div className="grid grid-cols-2 gap-2">
        {items.map((item) => (
          <div key={item.label} className="flex items-center gap-2 text-xs">
            <item.icon className="h-3 w-3 text-text-tertiary" />
            <span className="text-text-tertiary">{item.label}:</span>
            <span className="text-text-secondary truncate">{item.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function formatDate(timestamp: string | undefined): string {
  if (!timestamp) return '';
  return new Date(timestamp).toLocaleString();
}

function truncateId(id: string | undefined): string {
  if (!id) return '';
  return id.length > 12 ? `${id.slice(0, 6)}...${id.slice(-4)}` : id;
}
```

### Results List Component

```tsx
// src/components/results/ResultsList.tsx
import { ResultCard } from './ResultCard';
import { useSearchStore } from '@/stores/searchStore';

interface ResultsListProps {
  onSelect?: () => void;
}

export function ResultsList({ onSelect }: ResultsListProps) {
  const { results, query } = useSearchStore();

  if (results.length === 0) {
    return null;
  }

  return (
    <div className="space-y-2">
      <div className="text-xs text-text-tertiary px-1">
        {results.length} result{results.length !== 1 ? 's' : ''}
      </div>
      {results.map((result) => (
        <ResultCard
          key={result.id}
          result={result}
          query={query}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
}
```

## Acceptance Criteria

- [ ] Result cards display snippet, score, badges
- [ ] Query terms highlighted in snippets
- [ ] Entity type badges show correct icons/colors
- [ ] Source type badges display for OCR/audio
- [ ] Timestamps display in relative format
- [ ] Confidence indicator visualizes correctly
- [ ] Expand/collapse works for long content
- [ ] Quick actions visible on hover
- [ ] Copy action copies to clipboard
- [ ] Provenance panel toggles correctly
- [ ] Causal chain preview displays when present
- [ ] Results list shows count

## Test Plan

### Unit Tests
```typescript
describe('ResultCard', () => {
  it('should highlight query terms', () => {
    const content = 'Meeting with John about project';
    const result = highlightTerms(content, 'meeting');
    expect(result).toContain('<mark');
  });

  it('should format timestamp correctly', () => {
    const today = new Date().toISOString();
    expect(formatTimestamp(today)).toBe('Today');
  });
});
```

### E2E Tests
```typescript
test('result card interactions', async ({ page }) => {
  // Trigger search with results
  await page.keyboard.press('Meta+k');
  await page.fill('input', 'test query');
  await page.keyboard.press('Enter');

  // Wait for results
  const resultCard = page.locator('[data-testid="result-card"]').first();
  await expect(resultCard).toBeVisible();

  // Test expand
  await resultCard.locator('text=Show more').click();
  await expect(resultCard.locator('text=Show less')).toBeVisible();
});
```

## Dependencies

- Lucide React icons
- @/components/ui primitives

## Next Steps

After results view complete:
1. Proceed to Module 07 (Knowledge Graph Visualization)
2. Add result pinning/saving functionality
3. Implement result sharing

**This results view provides rich, actionable search results with full provenance transparency.**
