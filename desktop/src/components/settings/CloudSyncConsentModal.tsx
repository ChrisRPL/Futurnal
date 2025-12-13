/**
 * Cloud Sync Consent Modal
 *
 * Multi-step wizard for granting cloud sync consent.
 * Clearly explains what data will be synced and what won't.
 */

import { useState } from 'react';
import { Cloud, Shield, Check, X, AlertTriangle, ChevronRight, ChevronLeft } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { useCloudSyncStore } from '@/stores/cloudSyncStore';
import { useAuth } from '@/contexts/AuthContext';
import {
  CLOUD_SYNC_SCOPE_INFO,
  getDefaultEnabledScopes,
  type CloudSyncScope,
} from '@/types/api';

interface CloudSyncConsentModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

type Step = 'intro' | 'scopes' | 'confirm';

export function CloudSyncConsentModal({ open, onOpenChange }: CloudSyncConsentModalProps) {
  const [step, setStep] = useState<Step>('intro');
  const [selectedScopes, setSelectedScopes] = useState<CloudSyncScope[]>(getDefaultEnabledScopes());
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { grantConsent } = useCloudSyncStore();
  const { user } = useAuth();

  const handleClose = () => {
    setStep('intro');
    setSelectedScopes(getDefaultEnabledScopes());
    setError(null);
    onOpenChange(false);
  };

  const handleScopeToggle = (scope: CloudSyncScope, checked: boolean) => {
    const scopeInfo = CLOUD_SYNC_SCOPE_INFO.find((s) => s.scope === scope);

    // Don't allow unchecking required scopes
    if (scopeInfo?.required && !checked) {
      return;
    }

    if (checked) {
      setSelectedScopes((prev) => [...prev, scope]);
    } else {
      setSelectedScopes((prev) => prev.filter((s) => s !== scope));
    }
  };

  const handleConfirm = async () => {
    setIsSubmitting(true);
    setError(null);

    try {
      await grantConsent(selectedScopes, user?.email);
      handleClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to grant consent');
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderIntroStep = () => (
    <>
      <DialogHeader>
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 rounded-lg bg-[var(--color-surface)]">
            <Cloud className="h-6 w-6 text-[var(--color-text-secondary)]" />
          </div>
          <DialogTitle className="text-xl">Enable Cloud Sync</DialogTitle>
        </div>
        <DialogDescription className="text-[var(--color-text-tertiary)]">
          Sync your knowledge graph metadata to Firebase for access across devices.
        </DialogDescription>
      </DialogHeader>

      <div className="py-6 space-y-6">
        {/* What WILL be synced */}
        <div className="p-4 border border-[var(--color-border)] rounded-lg bg-[var(--color-surface)]">
          <div className="flex items-center gap-2 mb-3">
            <Check className="h-5 w-5 text-green-500" />
            <span className="font-medium text-[var(--color-text-primary)]">What will be synced</span>
          </div>
          <ul className="space-y-2 text-sm text-[var(--color-text-secondary)]">
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-0.5">-</span>
              Entity names (people, organizations, concepts, events)
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-0.5">-</span>
              Relationship types between entities
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-0.5">-</span>
              Timestamps (when entities were created/modified)
            </li>
            <li className="flex items-start gap-2">
              <span className="text-green-500 mt-0.5">-</span>
              Source identifiers (which connector created the data)
            </li>
          </ul>
        </div>

        {/* What will NOT be synced */}
        <div className="p-4 border border-red-900/30 rounded-lg bg-red-950/20">
          <div className="flex items-center gap-2 mb-3">
            <X className="h-5 w-5 text-red-500" />
            <span className="font-medium text-[var(--color-text-primary)]">What will NOT be synced</span>
          </div>
          <ul className="space-y-2 text-sm text-[var(--color-text-secondary)]">
            <li className="flex items-start gap-2">
              <span className="text-red-500 mt-0.5">-</span>
              Document content
            </li>
            <li className="flex items-start gap-2">
              <span className="text-red-500 mt-0.5">-</span>
              Email bodies
            </li>
            <li className="flex items-start gap-2">
              <span className="text-red-500 mt-0.5">-</span>
              File contents
            </li>
            <li className="flex items-start gap-2">
              <span className="text-red-500 mt-0.5">-</span>
              Attachment data
            </li>
          </ul>
        </div>

        {/* Privacy notice */}
        <div className="flex items-start gap-3 p-4 border border-[var(--color-border)] rounded-lg">
          <Shield className="h-5 w-5 text-[var(--color-text-tertiary)] mt-0.5 flex-shrink-0" />
          <div className="text-sm text-[var(--color-text-secondary)]">
            <strong className="text-[var(--color-text-primary)]">Privacy first:</strong> Your actual
            documents and content never leave your device. Only the graph structure is synced.
          </div>
        </div>
      </div>

      <DialogFooter>
        <Button variant="outline" onClick={handleClose}>
          Cancel
        </Button>
        <Button onClick={() => setStep('scopes')}>
          Continue
          <ChevronRight className="h-4 w-4 ml-1" />
        </Button>
      </DialogFooter>
    </>
  );

  const renderScopesStep = () => (
    <>
      <DialogHeader>
        <DialogTitle>Choose What to Sync</DialogTitle>
        <DialogDescription className="text-[var(--color-text-tertiary)]">
          Select which data categories you want to sync to the cloud.
        </DialogDescription>
      </DialogHeader>

      <div className="py-6 space-y-4">
        {CLOUD_SYNC_SCOPE_INFO.map((scopeInfo) => (
          <div
            key={scopeInfo.scope}
            className={`p-4 border rounded-lg transition-colors ${
              selectedScopes.includes(scopeInfo.scope)
                ? 'border-[var(--color-text-secondary)] bg-[var(--color-surface)]'
                : 'border-[var(--color-border)] bg-transparent'
            }`}
          >
            <div className="flex items-start gap-3">
              <Checkbox
                id={scopeInfo.scope}
                checked={selectedScopes.includes(scopeInfo.scope)}
                onCheckedChange={(checked) =>
                  handleScopeToggle(scopeInfo.scope, checked as boolean)
                }
                disabled={scopeInfo.required}
                className="mt-0.5"
              />
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <label
                    htmlFor={scopeInfo.scope}
                    className="text-sm font-medium text-[var(--color-text-primary)] cursor-pointer"
                  >
                    {scopeInfo.title}
                  </label>
                  {scopeInfo.required && (
                    <Badge variant="secondary" className="text-xs">
                      Required
                    </Badge>
                  )}
                </div>
                <p className="text-xs text-[var(--color-text-tertiary)] mt-1">
                  {scopeInfo.description}
                </p>

                {/* Data shared details */}
                {selectedScopes.includes(scopeInfo.scope) && scopeInfo.data_shared && (
                  <div className="mt-3 pt-3 border-t border-[var(--color-border)]">
                    <div className="text-xs text-[var(--color-text-muted)] mb-2">Data shared:</div>
                    <div className="flex flex-wrap gap-1">
                      {scopeInfo.data_shared.map((item, idx) => (
                        <Badge key={idx} variant="outline" className="text-xs">
                          {item}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      <DialogFooter>
        <Button variant="outline" onClick={() => setStep('intro')}>
          <ChevronLeft className="h-4 w-4 mr-1" />
          Back
        </Button>
        <Button onClick={() => setStep('confirm')}>
          Review & Confirm
          <ChevronRight className="h-4 w-4 ml-1" />
        </Button>
      </DialogFooter>
    </>
  );

  const renderConfirmStep = () => (
    <>
      <DialogHeader>
        <DialogTitle>Confirm Cloud Sync</DialogTitle>
        <DialogDescription className="text-[var(--color-text-tertiary)]">
          Review your selections before enabling cloud sync.
        </DialogDescription>
      </DialogHeader>

      <div className="py-6 space-y-6">
        {/* Selected scopes summary */}
        <div className="p-4 border border-[var(--color-border)] rounded-lg bg-[var(--color-surface)]">
          <div className="text-sm font-medium text-[var(--color-text-primary)] mb-3">
            You are enabling sync for:
          </div>
          <div className="space-y-2">
            {CLOUD_SYNC_SCOPE_INFO.filter((s) => selectedScopes.includes(s.scope)).map(
              (scopeInfo) => (
                <div key={scopeInfo.scope} className="flex items-center gap-2">
                  <Check className="h-4 w-4 text-green-500" />
                  <span className="text-sm text-[var(--color-text-secondary)]">
                    {scopeInfo.title}
                  </span>
                </div>
              )
            )}
          </div>
        </div>

        <Separator />

        {/* Important notices */}
        <div className="space-y-3">
          <div className="flex items-start gap-3 text-sm">
            <Check className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
            <span className="text-[var(--color-text-secondary)]">
              Sync will happen every 15 minutes while the app is open
            </span>
          </div>
          <div className="flex items-start gap-3 text-sm">
            <Check className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
            <span className="text-[var(--color-text-secondary)]">
              You can revoke consent at any time from Privacy settings
            </span>
          </div>
          <div className="flex items-start gap-3 text-sm">
            <AlertTriangle className="h-4 w-4 text-amber-500 mt-0.5 flex-shrink-0" />
            <span className="text-[var(--color-text-secondary)]">
              Revoking consent will <strong>delete all your cloud data</strong>
            </span>
          </div>
        </div>

        {error && (
          <div className="p-3 border border-red-900/50 rounded-lg bg-red-950/30 text-red-400 text-sm">
            {error}
          </div>
        )}
      </div>

      <DialogFooter>
        <Button variant="outline" onClick={() => setStep('scopes')} disabled={isSubmitting}>
          <ChevronLeft className="h-4 w-4 mr-1" />
          Back
        </Button>
        <Button onClick={handleConfirm} disabled={isSubmitting}>
          {isSubmitting ? 'Enabling...' : 'Enable Cloud Sync'}
        </Button>
      </DialogFooter>
    </>
  );

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[500px]">
        {step === 'intro' && renderIntroStep()}
        {step === 'scopes' && renderScopesStep()}
        {step === 'confirm' && renderConfirmStep()}
      </DialogContent>
    </Dialog>
  );
}
