/**
 * UpgradePrompt Component
 *
 * Modal dialog shown when free tier users hit feature limits.
 * Uses existing Dialog primitive, follows Futurnal monochrome design system.
 */

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import { openUpgradePage } from '@/lib/subscription';
import { Crown } from 'lucide-react';

interface UpgradePromptProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  reason: string;
}

function FeatureItem({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-3 text-sm text-white/70">
      <div className="w-1 h-1 bg-white/40" />
      <span>{children}</span>
    </div>
  );
}

export function UpgradePrompt({ open, onOpenChange, reason }: UpgradePromptProps) {
  const handleUpgrade = async () => {
    await openUpgradePage();
    onOpenChange(false);
  };

  const handleClose = () => {
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-black border-white/10 max-w-md">
        <DialogHeader className="text-center sm:text-center">
          {/* Crown icon */}
          <div className="mx-auto w-12 h-12 flex items-center justify-center border border-white/20 mb-4">
            <Crown className="w-6 h-6 text-white/60" />
          </div>

          <DialogTitle className="font-brand text-2xl tracking-wide text-white">
            Upgrade to Pro
          </DialogTitle>

          <DialogDescription className="text-white/60 mt-2">{reason}</DialogDescription>
        </DialogHeader>

        {/* Feature highlights */}
        <div className="py-4 space-y-3">
          <FeatureItem>Unlimited data sources</FeatureItem>
          <FeatureItem>Cloud backup & sync</FeatureItem>
          <FeatureItem>Emergent insights (Phase 2)</FeatureItem>
          <FeatureItem>Causal exploration (Phase 3)</FeatureItem>
        </div>

        <DialogFooter className="flex flex-col sm:flex-row gap-3 sm:gap-4">
          <button
            onClick={handleClose}
            className="flex-1 py-3 border border-white/30 text-white/70 hover:border-white/60 transition-all"
          >
            Maybe Later
          </button>
          <button
            onClick={handleUpgrade}
            className="flex-1 py-3 bg-white text-black font-medium hover:bg-white/90 transition-all"
          >
            View Plans
          </button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default UpgradePrompt;
