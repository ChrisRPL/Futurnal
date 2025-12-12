/**
 * AddConnectorModal Component
 *
 * Two-step wizard for adding new data source connectors.
 * Step 1: Select connector type
 * Step 2: Configure connector (type-specific form)
 */

import { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Folder, FileText, Github, Mail, ArrowLeft, Loader2, CheckCircle2, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useAddConnector } from '@/hooks/useApi';
import { open as openFolderDialog } from '@tauri-apps/plugin-dialog';
import { invoke } from '@tauri-apps/api/core';
import type { ConnectorType } from '@/types/api';

// Provider configuration for auto-detection
const PROVIDER_CONFIG: Record<string, { server: string; port: number; name: string }> = {
  'gmail.com': { server: 'imap.gmail.com', port: 993, name: 'Gmail' },
  'googlemail.com': { server: 'imap.gmail.com', port: 993, name: 'Gmail' },
  'outlook.com': { server: 'outlook.office365.com', port: 993, name: 'Outlook' },
  'hotmail.com': { server: 'outlook.office365.com', port: 993, name: 'Outlook' },
  'live.com': { server: 'outlook.office365.com', port: 993, name: 'Outlook' },
  'yahoo.com': { server: 'imap.mail.yahoo.com', port: 993, name: 'Yahoo' },
  'icloud.com': { server: 'imap.mail.me.com', port: 993, name: 'iCloud' },
  'me.com': { server: 'imap.mail.me.com', port: 993, name: 'iCloud' },
  'mac.com': { server: 'imap.mail.me.com', port: 993, name: 'iCloud' },
  'protonmail.com': { server: 'imap.protonmail.ch', port: 993, name: 'ProtonMail' },
  'proton.me': { server: 'imap.protonmail.ch', port: 993, name: 'ProtonMail' },
};

// Get provider config from email domain
const getProviderFromEmail = (email: string) => {
  const domain = email.split('@')[1]?.toLowerCase();
  return domain ? PROVIDER_CONFIG[domain] || null : null;
};

// Get contextual password help based on provider
const getPasswordHelp = (email: string): { text: string; url?: string } => {
  const domain = email?.split('@')[1]?.toLowerCase();
  switch (domain) {
    case 'gmail.com':
    case 'googlemail.com':
      return {
        text: 'Create an App Password (requires 2FA enabled)',
        url: 'https://myaccount.google.com/apppasswords',
      };
    case 'outlook.com':
    case 'hotmail.com':
    case 'live.com':
      return {
        text: 'Create an App Password in your Microsoft account',
        url: 'https://account.live.com/proofs/AppPassword',
      };
    case 'yahoo.com':
      return {
        text: 'Generate an App Password in Yahoo Account Security',
        url: 'https://login.yahoo.com/myaccount/security/app-password',
      };
    case 'icloud.com':
    case 'me.com':
    case 'mac.com':
      return {
        text: 'Generate an App-Specific Password',
        url: 'https://appleid.apple.com/account/manage',
      };
    case 'protonmail.com':
    case 'proton.me':
      return {
        text: 'ProtonMail requires Bridge app for IMAP access',
        url: 'https://proton.me/mail/bridge',
      };
    default:
      return {
        text: 'Enter your email password or App Password if required by your provider.',
      };
  }
};

// Connection test result type
interface ConnectionTestResult {
  success: boolean;
  message?: string;
  error?: string;
  folders?: number;
}

interface AddConnectorModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const CONNECTOR_TYPES = [
  {
    type: 'local_folder' as ConnectorType,
    label: 'Local Folder',
    description: 'Index documents, images, and files from your computer',
    icon: Folder,
  },
  {
    type: 'obsidian' as ConnectorType,
    label: 'Obsidian Vault',
    description: 'Connect your Obsidian notes and knowledge base',
    icon: FileText,
  },
  {
    type: 'github' as ConnectorType,
    label: 'GitHub Repository',
    description: 'Import code, issues, and documentation from GitHub',
    icon: Github,
  },
  {
    type: 'imap' as ConnectorType,
    label: 'Email (IMAP)',
    description: 'Index emails and communication history',
    icon: Mail,
  },
];

export function AddConnectorModal({ open, onOpenChange }: AddConnectorModalProps) {
  const [step, setStep] = useState<'select' | 'configure'>('select');
  const [selectedType, setSelectedType] = useState<ConnectorType | null>(null);
  const [config, setConfig] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<ConnectionTestResult | null>(null);

  const addMutation = useAddConnector();

  // Handle email change with auto-fill for server
  const handleEmailChange = (email: string) => {
    setConfig((prev) => {
      const newConfig: Record<string, string> = { ...prev, email };
      const provider = getProviderFromEmail(email);
      if (provider && !prev.server) {
        newConfig.server = provider.server;
      }
      return newConfig;
    });
    // Clear test result when email changes
    setTestResult(null);
  };

  const handleSelectType = (type: ConnectorType) => {
    setSelectedType(type);
    setStep('configure');
    setConfig({});
    setError(null);
  };

  const handleBack = () => {
    setStep('select');
    setSelectedType(null);
    setConfig({});
    setError(null);
    setTestResult(null);
  };

  const handleClose = () => {
    onOpenChange(false);
    // Reset state after animation
    setTimeout(() => {
      setStep('select');
      setSelectedType(null);
      setConfig({});
      setError(null);
      setTestResult(null);
    }, 200);
  };

  const handleBrowseFolder = async () => {
    try {
      const selected = await openFolderDialog({
        directory: true,
        multiple: false,
        title: selectedType === 'obsidian' ? 'Select Obsidian Vault' : 'Select Folder',
      });
      if (selected) {
        setConfig((prev) => ({ ...prev, path: selected as string }));
      }
    } catch (err) {
      console.error('Failed to open folder picker:', err);
    }
  };

  const handleSubmit = async () => {
    if (!selectedType) return;

    // Validate required fields
    if (!config.name?.trim()) {
      setError('Name is required');
      return;
    }

    if ((selectedType === 'local_folder' || selectedType === 'obsidian') && !config.path?.trim()) {
      setError('Folder path is required');
      return;
    }

    if (selectedType === 'github' && !config.repo?.trim()) {
      setError('Repository URL is required');
      return;
    }

    if (selectedType === 'imap') {
      if (!config.server?.trim()) {
        setError('IMAP server is required');
        return;
      }
      if (!config.email?.trim()) {
        setError('Email address is required');
        return;
      }
      if (!config.password?.trim()) {
        setError('Password is required');
        return;
      }

      // Test IMAP connection before adding
      setIsTesting(true);
      setError(null);
      try {
        const result = await invoke<ConnectionTestResult>('test_imap_connection', {
          email: config.email.trim(),
          server: config.server.trim(),
          password: config.password,
        });
        setTestResult(result);
        if (!result.success) {
          setError(result.error || 'Connection failed. Check your credentials.');
          setIsTesting(false);
          return;
        }
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'Connection test failed';
        setError(errorMsg);
        setTestResult({ success: false, error: errorMsg });
        setIsTesting(false);
        return;
      }
      setIsTesting(false);
    }

    setError(null);

    try {
      await addMutation.mutateAsync({
        connector_type: selectedType,
        name: config.name.trim(),
        config: {
          ...config,
          name: undefined, // Remove name from config, it's a top-level field
        },
      });
      handleClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add data source');
    }
  };

  const isLoading = addMutation.isPending || isTesting;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg bg-black border-white/10">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-white">
            {step === 'configure' && (
              <button
                onClick={handleBack}
                className="hover:bg-white/10 p-1 transition-colors"
                disabled={isLoading}
              >
                <ArrowLeft className="h-4 w-4" />
              </button>
            )}
            {step === 'select' ? 'Add Data Source' : `Configure ${CONNECTOR_TYPES.find(t => t.type === selectedType)?.label}`}
          </DialogTitle>
        </DialogHeader>

        {step === 'select' ? (
          <div className="grid grid-cols-2 gap-3 py-4">
            {CONNECTOR_TYPES.map((type) => (
              <button
                key={type.type}
                onClick={() => handleSelectType(type.type)}
                className={cn(
                  'flex flex-col items-start p-4 border border-white/10',
                  'hover:border-white/30 hover:bg-white/5 transition-colors',
                  'text-left'
                )}
              >
                <type.icon className="h-6 w-6 text-white/60 mb-2" />
                <div className="font-medium text-white">{type.label}</div>
                <div className="text-xs text-white/50 mt-1">
                  {type.description}
                </div>
              </button>
            ))}
          </div>
        ) : (
          <div className="py-4 space-y-4">
            {/* Error Display */}
            {error && (
              <div className="p-3 bg-red-500/10 border border-red-500/20 text-sm text-red-400">
                {error}
              </div>
            )}

            {/* Name */}
            <div>
              <Label htmlFor="name">Name</Label>
              <Input
                id="name"
                value={config.name || ''}
                onChange={(e) => setConfig((prev) => ({ ...prev, name: e.target.value }))}
                placeholder="My Data Source"
                className="mt-1"
                disabled={isLoading}
              />
            </div>

            {/* Type-specific fields */}
            {(selectedType === 'local_folder' || selectedType === 'obsidian') && (
              <div>
                <Label htmlFor="path">
                  {selectedType === 'obsidian' ? 'Vault Path' : 'Folder Path'}
                </Label>
                <div className="flex gap-2 mt-1">
                  <Input
                    id="path"
                    value={config.path || ''}
                    onChange={(e) => setConfig((prev) => ({ ...prev, path: e.target.value }))}
                    placeholder="/path/to/folder"
                    className="flex-1"
                    disabled={isLoading}
                  />
                  <Button
                    type="button"
                    variant="outline"
                    onClick={handleBrowseFolder}
                    className="border-white/20 text-white/70 hover:bg-white/10"
                    disabled={isLoading}
                  >
                    Browse
                  </Button>
                </div>
              </div>
            )}

            {selectedType === 'github' && (
              <>
                <div>
                  <Label htmlFor="repo">Repository</Label>
                  <Input
                    id="repo"
                    value={config.repo || ''}
                    onChange={(e) => setConfig((prev) => ({ ...prev, repo: e.target.value }))}
                    placeholder="owner/repo"
                    className="mt-1"
                    disabled={isLoading}
                  />
                  <p className="text-xs text-white/40 mt-1">
                    Enter in format: owner/repo (e.g., anthropics/claude-code)
                  </p>
                </div>
                <div>
                  <Label htmlFor="token">Personal Access Token</Label>
                  <Input
                    id="token"
                    type="password"
                    value={config.token || ''}
                    onChange={(e) => setConfig((prev) => ({ ...prev, token: e.target.value }))}
                    placeholder="ghp_..."
                    className="mt-1"
                    disabled={isLoading}
                  />
                  <p className="text-xs text-white/40 mt-1">
                    Required for private repos. Create at GitHub Settings → Developer Settings.
                  </p>
                </div>
              </>
            )}

            {selectedType === 'imap' && (
              <>
                <div>
                  <Label htmlFor="email">Email Address</Label>
                  <Input
                    id="email"
                    type="email"
                    value={config.email || ''}
                    onChange={(e) => handleEmailChange(e.target.value)}
                    placeholder="you@example.com"
                    className="mt-1"
                    disabled={isLoading}
                  />
                  {config.email && getProviderFromEmail(config.email) && (
                    <p className="text-xs text-green-400/70 mt-1">
                      {getProviderFromEmail(config.email)?.name} detected - server auto-filled
                    </p>
                  )}
                </div>
                <div>
                  <Label htmlFor="server">IMAP Server</Label>
                  <Input
                    id="server"
                    value={config.server || ''}
                    onChange={(e) => {
                      setConfig((prev) => ({ ...prev, server: e.target.value }));
                      setTestResult(null);
                    }}
                    placeholder="imap.gmail.com"
                    className="mt-1"
                    disabled={isLoading}
                  />
                </div>
                <div>
                  <Label htmlFor="password">Password / App Password</Label>
                  <Input
                    id="password"
                    type="password"
                    value={config.password || ''}
                    onChange={(e) => {
                      setConfig((prev) => ({ ...prev, password: e.target.value }));
                      setTestResult(null);
                    }}
                    className="mt-1"
                    disabled={isLoading}
                  />
                  {(() => {
                    const help = getPasswordHelp(config.email || '');
                    return (
                      <p className="text-xs text-white/40 mt-1">
                        {help.url ? (
                          <>
                            {help.text}{' '}
                            <a
                              href={help.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="underline hover:text-white/60"
                            >
                              Get one here
                            </a>
                          </>
                        ) : (
                          help.text
                        )}
                      </p>
                    );
                  })()}
                </div>
                {/* Connection test result */}
                {testResult && (
                  <div className={cn(
                    "p-3 border text-sm flex items-center gap-2",
                    testResult.success
                      ? "bg-green-500/10 border-green-500/20 text-green-400"
                      : "bg-red-500/10 border-red-500/20 text-red-400"
                  )}>
                    {testResult.success ? (
                      <>
                        <CheckCircle2 className="h-4 w-4" />
                        Connection successful{testResult.folders ? ` (${testResult.folders} folders found)` : ''}
                      </>
                    ) : (
                      <>
                        <XCircle className="h-4 w-4" />
                        {testResult.error || 'Connection failed'}
                      </>
                    )}
                  </div>
                )}
              </>
            )}

            {/* Submit */}
            <div className="flex justify-end gap-2 pt-4">
              <Button
                variant="outline"
                onClick={handleClose}
                className="border-white/20 text-white/70 hover:bg-white/10"
                disabled={isLoading}
              >
                Cancel
              </Button>
              <Button
                onClick={handleSubmit}
                disabled={isLoading}
                className="bg-white text-black hover:bg-white/90"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    {isTesting ? 'Testing Connection...' : 'Adding...'}
                  </>
                ) : (
                  'Add Source'
                )}
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

export default AddConnectorModal;
