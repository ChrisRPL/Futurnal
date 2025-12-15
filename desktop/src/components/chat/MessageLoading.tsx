/**
 * MessageLoading - Animated typing indicator for chat
 *
 * Step 03: Chat Interface & Conversational AI
 *
 * Shows bouncing dots animation while AI is generating response.
 */

import { Bot } from 'lucide-react';
import { cn } from '@/lib/utils';

interface MessageLoadingProps {
  className?: string;
}

/**
 * Bouncing dots animation for typing indicator
 */
function BouncingDots() {
  return (
    <div className="flex items-center gap-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-2 h-2 bg-white/40 rounded-full animate-bounce"
          style={{
            animationDelay: `${i * 0.15}s`,
            animationDuration: '0.6s',
          }}
        />
      ))}
    </div>
  );
}

/**
 * MessageLoading component - shows while AI is thinking
 */
export function MessageLoading({ className }: MessageLoadingProps) {
  return (
    <div className={cn('flex gap-3', className)}>
      {/* Avatar */}
      <div className="flex-shrink-0 w-8 h-8 flex items-center justify-center bg-white/10 text-white/80 border border-white/20">
        <Bot className="h-4 w-4" />
      </div>

      {/* Loading bubble */}
      <div className="bg-white/5 border border-white/10 p-3">
        <BouncingDots />
      </div>
    </div>
  );
}
