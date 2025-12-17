/**
 * CausalVerification - Interactive causal hypothesis verification UI
 *
 * AGI Phase 8: Frontend Integration
 *
 * Research Foundation:
 * - ICDA (2024): Interactive Causal Discovery Agent
 * - ACCESS (2025): Causal validation metrics
 * - Human-in-the-loop causal inference
 *
 * Features:
 * - Question display with evidence
 * - Response option selection
 * - Optional explanation input
 * - Confidence update feedback
 */

import { useState } from 'react';
import {
  GitBranch,
  HelpCircle,
  ChevronRight,
  MessageSquare,
  Check,
  X,
  RefreshCw,
  AlertCircle,
  Loader2,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import type {
  CausalVerificationQuestion,
  CausalResponseType,
} from '@/stores/insightsStore';

interface CausalVerificationProps {
  question: CausalVerificationQuestion;
  onSubmit: (
    questionId: string,
    response: CausalResponseType,
    explanation?: string
  ) => Promise<{ success: boolean; confidenceDelta?: number } | null>;
  onSkip?: (questionId: string) => void;
  className?: string;
}

interface VerificationResult {
  success: boolean;
  confidenceDelta: number;
}

/** Response type descriptions */
const RESPONSE_DESCRIPTIONS: Record<CausalResponseType, string> = {
  yes_causal: 'I believe there is a causal relationship',
  no_correlation: 'They just happen together by coincidence',
  reverse_causation: 'The causation goes the other way',
  confounder: 'A third factor causes both',
  uncertain: "I'm not sure about this relationship",
  skip: 'Skip this question for now',
};

/** Response icons */
const RESPONSE_ICONS: Record<CausalResponseType, typeof Check> = {
  yes_causal: Check,
  no_correlation: X,
  reverse_causation: RefreshCw,
  confounder: GitBranch,
  uncertain: HelpCircle,
  skip: ChevronRight,
};

/** Response colors */
const RESPONSE_COLORS: Record<CausalResponseType, string> = {
  yes_causal: 'border-green-500/30 hover:bg-green-500/10',
  no_correlation: 'border-red-500/30 hover:bg-red-500/10',
  reverse_causation: 'border-amber-500/30 hover:bg-amber-500/10',
  confounder: 'border-purple-500/30 hover:bg-purple-500/10',
  uncertain: 'border-white/20 hover:bg-white/5',
  skip: 'border-white/10 hover:bg-white/5',
};

/**
 * Confidence indicator
 */
function ConfidenceIndicator({
  initial,
  delta,
  className,
}: {
  initial: number;
  delta?: number;
  className?: string;
}) {
  const initialPercent = Math.round(initial * 100);
  const newPercent = delta !== undefined ? Math.round((initial + delta) * 100) : null;

  return (
    <div className={cn('flex items-center gap-2', className)}>
      <span className="text-[10px] text-white/40">Confidence:</span>
      <span className="text-xs text-white/70">{initialPercent}%</span>
      {delta !== undefined && (
        <>
          <ChevronRight className="w-3 h-3 text-white/30" />
          <span
            className={cn(
              'text-xs font-medium',
              delta > 0 ? 'text-green-400' : delta < 0 ? 'text-red-400' : 'text-white/70'
            )}
          >
            {newPercent}%
            <span className="ml-1 text-[10px]">
              ({delta > 0 ? '+' : ''}
              {Math.round(delta * 100)}%)
            </span>
          </span>
        </>
      )}
    </div>
  );
}

/**
 * Main CausalVerification component
 */
export function CausalVerification({
  question,
  onSubmit,
  onSkip,
  className,
}: CausalVerificationProps) {
  const [selectedResponse, setSelectedResponse] = useState<CausalResponseType | null>(
    null
  );
  const [explanation, setExplanation] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [showExplanation, setShowExplanation] = useState(false);

  const handleSubmit = async () => {
    if (!selectedResponse) return;

    setIsSubmitting(true);
    const response = await onSubmit(
      question.questionId,
      selectedResponse,
      explanation.trim() || undefined
    );
    setIsSubmitting(false);

    if (response?.success) {
      setResult({
        success: true,
        confidenceDelta: response.confidenceDelta || 0,
      });
    }
  };

  const handleSkip = () => {
    onSkip?.(question.questionId);
  };

  // Show result state
  if (result) {
    return (
      <div
        className={cn(
          'bg-white/[0.02] border border-white/10 p-4',
          className
        )}
      >
        <div className="flex items-center justify-center gap-2 text-green-400">
          <Check className="w-5 h-5" />
          <span className="text-sm font-medium">Response Recorded</span>
        </div>
        <div className="mt-3 text-center">
          <ConfidenceIndicator
            initial={question.initialConfidence}
            delta={result.confidenceDelta}
            className="justify-center"
          />
        </div>
        <p className="mt-3 text-xs text-white/50 text-center">
          Your feedback helps improve causal understanding.
        </p>
      </div>
    );
  }

  return (
    <div className={cn('bg-white/[0.02] border border-white/10', className)}>
      {/* Header */}
      <div className="p-4 border-b border-white/5">
        <div className="flex items-center gap-2 mb-2">
          <GitBranch className="w-4 h-4 text-purple-400" />
          <span className="text-xs text-white/40">Causal Verification</span>
        </div>

        {/* Main question */}
        <p className="text-sm text-white/90 leading-relaxed">
          {question.mainQuestion}
        </p>

        {/* Context */}
        <p className="mt-2 text-xs text-white/50">{question.context}</p>

        {/* Evidence */}
        <div className="mt-3 p-2 bg-white/[0.02] border border-white/5">
          <p className="text-[10px] text-white/40 mb-1">Evidence Summary</p>
          <pre className="text-[10px] text-white/60 whitespace-pre-wrap font-mono">
            {question.evidenceSummary}
          </pre>
        </div>

        {/* Initial confidence */}
        <div className="mt-3">
          <ConfidenceIndicator initial={question.initialConfidence} />
        </div>
      </div>

      {/* Response options */}
      <div className="p-4 space-y-2">
        <p className="text-xs text-white/40 mb-2">Select your response:</p>

        {question.responseOptions.map((option) => {
          const Icon = RESPONSE_ICONS[option.value] || HelpCircle;
          const colorClass = RESPONSE_COLORS[option.value];
          const isSelected = selectedResponse === option.value;

          return (
            <button
              key={option.value}
              onClick={() => {
                setSelectedResponse(option.value);
                // Show explanation for non-skip responses
                setShowExplanation(option.value !== 'skip');
              }}
              className={cn(
                'flex items-center gap-3 w-full text-left',
                'p-3 border transition-all',
                colorClass,
                isSelected
                  ? 'bg-white/10 border-white/30'
                  : 'border-white/10 bg-transparent'
              )}
            >
              <Icon
                className={cn(
                  'w-4 h-4 flex-shrink-0',
                  isSelected ? 'text-white' : 'text-white/40'
                )}
              />
              <div className="flex-1 min-w-0">
                <p
                  className={cn(
                    'text-sm',
                    isSelected ? 'text-white' : 'text-white/70'
                  )}
                >
                  {option.label}
                </p>
                <p className="text-[10px] text-white/40 mt-0.5">
                  {RESPONSE_DESCRIPTIONS[option.value]}
                </p>
              </div>
              {isSelected && <Check className="w-4 h-4 text-white/60" />}
            </button>
          );
        })}

        {/* Explanation input */}
        {showExplanation && selectedResponse && selectedResponse !== 'skip' && (
          <div className="mt-3">
            <div className="flex items-center gap-2 text-xs text-white/40 mb-1">
              <MessageSquare className="w-3 h-3" />
              <span>Optional: Explain your reasoning</span>
            </div>
            <textarea
              value={explanation}
              onChange={(e) => setExplanation(e.target.value)}
              placeholder="Your explanation helps refine causal understanding..."
              className={cn(
                'w-full h-20 px-3 py-2',
                'bg-white/[0.02] border border-white/10',
                'text-sm text-white/80 placeholder:text-white/30',
                'focus:outline-none focus:border-white/20',
                'resize-none'
              )}
            />
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="p-4 border-t border-white/5 flex items-center justify-between gap-3">
        <button
          onClick={handleSkip}
          className="text-xs text-white/40 hover:text-white/60 transition-colors"
        >
          Skip for now
        </button>

        <button
          onClick={handleSubmit}
          disabled={!selectedResponse || isSubmitting}
          className={cn(
            'flex items-center gap-2 px-4 py-2',
            'bg-white/10 hover:bg-white/20',
            'text-sm text-white/80 hover:text-white',
            'transition-colors',
            (!selectedResponse || isSubmitting) && 'opacity-50 cursor-not-allowed'
          )}
        >
          {isSubmitting ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Submitting...</span>
            </>
          ) : (
            <>
              <Check className="w-4 h-4" />
              <span>Submit Response</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}

/**
 * Compact verification card for feeds
 */
export function CausalVerificationCompact({
  question,
  onClick,
  className,
}: {
  question: CausalVerificationQuestion;
  onClick?: () => void;
  className?: string;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'w-full text-left p-3',
        'bg-white/[0.02] border border-white/10',
        'hover:bg-white/[0.04] transition-colors',
        className
      )}
    >
      <div className="flex items-start gap-3">
        <div className="p-1.5 rounded bg-purple-500/20">
          <GitBranch className="w-4 h-4 text-purple-400" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs text-white/40 mb-1">Verify Causal Relationship</p>
          <p className="text-sm text-white/80 line-clamp-2">
            Does <span className="text-purple-400">{question.causeEvent}</span> cause{' '}
            <span className="text-purple-400">{question.effectEvent}</span>?
          </p>
          <div className="mt-2 flex items-center gap-2">
            <AlertCircle className="w-3 h-3 text-amber-400" />
            <span className="text-[10px] text-amber-400">Needs Verification</span>
            <span className="text-[10px] text-white/30">
              {Math.round(question.initialConfidence * 100)}% confidence
            </span>
          </div>
        </div>
        <ChevronRight className="w-4 h-4 text-white/30 flex-shrink-0" />
      </div>
    </button>
  );
}

export default CausalVerification;
