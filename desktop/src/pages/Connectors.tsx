/**
 * Connectors Page - Data Source Management
 *
 * Dedicated route (/connectors) for managing data source connectors.
 * Shows list of connectors, status, controls, and add/settings modals.
 */

import { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Database, Plus, RefreshCw, Pause, Play, AlertTriangle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ConnectorCard, AddConnectorModal, ConnectorSettings } from '@/components/connectors';
import { UpgradePrompt } from '@/components/UpgradePrompt';
import {
  useConnectors,
  usePauseAllConnectors,
  useResumeAllConnectors,
} from '@/hooks/useApi';
import { useTierLimits } from '@/hooks/useTierLimits';
import { useQueryClient } from '@tanstack/react-query';
import { queryKeys } from '@/lib/queryClient';
import type { Connector } from '@/types/api';

export function ConnectorsPage() {
  const queryClient = useQueryClient();

  // Data
  const { data: connectors, isLoading, isRefetching } = useConnectors();
  const { canAddDataSource, isPro, getUpgradeReason } = useTierLimits();

  // Mutations
  const pauseAllMutation = usePauseAllConnectors();
  const resumeAllMutation = useResumeAllConnectors();

  // Modal state
  const [addModalOpen, setAddModalOpen] = useState(false);
  const [settingsConnector, setSettingsConnector] = useState<Connector | null>(null);
  const [upgradeOpen, setUpgradeOpen] = useState(false);

  // Computed
  const activeCount = connectors?.filter(
    (c) => c.status === 'active' || c.status === 'syncing'
  ).length ?? 0;
  const pausedCount = connectors?.filter((c) => c.status === 'paused').length ?? 0;
  const errorCount = connectors?.filter((c) => c.status === 'error').length ?? 0;
  const totalCount = connectors?.length ?? 0;

  const canAdd = canAddDataSource(totalCount);
  const isAtLimit = !isPro && totalCount >= 3;
  const isMutating = pauseAllMutation.isPending || resumeAllMutation.isPending;

  // Handlers
  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: queryKeys.connectors });
  };

  const handleAddSource = () => {
    if (canAdd) {
      setAddModalOpen(true);
    } else {
      setUpgradeOpen(true);
    }
  };

  const handlePauseAll = () => {
    pauseAllMutation.mutate();
  };

  const handleResumeAll = () => {
    resumeAllMutation.mutate();
  };

  const handleOpenSettings = (connector: Connector) => {
    setSettingsConnector(connector);
  };

  return (
    <div className="min-h-screen bg-black flex flex-col">
      {/* Header */}
      <header className="flex-shrink-0 border-b border-white/10">
        <div className="flex items-center justify-between px-6 py-4">
          {/* Left: Back + Title */}
          <div className="flex items-center gap-4">
            <Link to="/dashboard">
              <Button
                variant="ghost"
                size="icon"
                className="h-9 w-9 text-white/60 hover:text-white hover:bg-white/10"
              >
                <ArrowLeft className="h-5 w-5" />
                <span className="sr-only">Back to Dashboard</span>
              </Button>
            </Link>
            <div className="flex items-center gap-3">
              <Database className="h-5 w-5 text-white/60" />
              <h1 className="text-lg font-brand tracking-wide text-white">
                Data Sources
              </h1>
              {totalCount > 0 && (
                <Badge variant="secondary" className="text-xs">
                  {totalCount}
                </Badge>
              )}
            </div>
          </div>

          {/* Right: Actions */}
          <div className="flex items-center gap-3">
            {/* Refresh */}
            <Button
              variant="ghost"
              size="icon"
              onClick={handleRefresh}
              disabled={isRefetching}
              className="h-9 w-9 text-white/60 hover:text-white hover:bg-white/10"
            >
              <RefreshCw
                className={`h-4 w-4 ${isRefetching ? 'animate-spin' : ''}`}
              />
              <span className="sr-only">Refresh</span>
            </Button>

            {/* Pause All - show when any active */}
            {activeCount > 0 && (
              <Button
                variant="outline"
                size="sm"
                onClick={handlePauseAll}
                disabled={isMutating}
                className="border-white/20 text-white/70 hover:bg-white/10"
              >
                <Pause className="h-4 w-4 mr-2" />
                Pause All
              </Button>
            )}

            {/* Resume All - show when any paused */}
            {pausedCount > 0 && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleResumeAll}
                disabled={isMutating}
                className="border-white/20 text-white/70 hover:bg-white/10"
              >
                <Play className="h-4 w-4 mr-2" />
                Resume All
              </Button>
            )}

            {/* Add Source */}
            <Button
              onClick={handleAddSource}
              className="bg-white text-black hover:bg-white/90"
            >
              <Plus className="h-4 w-4 mr-2" />
              Add Source
            </Button>
          </div>
        </div>
      </header>

      {/* Tier Warning */}
      {isAtLimit && (
        <div className="px-6 py-3 bg-white/5 border-b border-white/10">
          <div className="flex items-center gap-3 text-sm">
            <AlertTriangle className="h-4 w-4 text-yellow-500" />
            <span className="text-white/70">
              You've reached the free tier limit of 3 data sources.
            </span>
            <button
              onClick={() => setUpgradeOpen(true)}
              className="text-white underline underline-offset-2 hover:no-underline"
            >
              Upgrade to Pro
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <ScrollArea className="flex-1">
        <div className="p-6 max-w-4xl mx-auto">
          {isLoading ? (
            // Loading state
            <div className="space-y-4">
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="h-24 bg-white/5 border border-white/10 animate-pulse"
                />
              ))}
            </div>
          ) : !connectors || connectors.length === 0 ? (
            // Empty state
            <div className="text-center py-20">
              <div className="w-16 h-16 mx-auto mb-6 flex items-center justify-center border border-white/20">
                <Database className="h-8 w-8 text-white/40" />
              </div>
              <h2 className="text-xl font-brand tracking-wide text-white mb-2">
                No data sources connected
              </h2>
              <p className="text-white/50 mb-8 max-w-md mx-auto">
                Connect your first data source to start building your personal knowledge graph.
              </p>
              <Button
                onClick={handleAddSource}
                className="bg-white text-black hover:bg-white/90"
              >
                <Plus className="h-4 w-4 mr-2" />
                Add Data Source
              </Button>
            </div>
          ) : (
            // Connector list
            <div className="space-y-4">
              {/* Status summary */}
              {(errorCount > 0 || activeCount > 0 || pausedCount > 0) && (
                <div className="flex items-center gap-4 text-sm text-white/50 mb-6">
                  {activeCount > 0 && (
                    <span className="flex items-center gap-2">
                      <span className="w-2 h-2 bg-green-500" />
                      {activeCount} active
                    </span>
                  )}
                  {pausedCount > 0 && (
                    <span className="flex items-center gap-2">
                      <span className="w-2 h-2 bg-yellow-500" />
                      {pausedCount} paused
                    </span>
                  )}
                  {errorCount > 0 && (
                    <span className="flex items-center gap-2">
                      <span className="w-2 h-2 bg-red-500" />
                      {errorCount} with errors
                    </span>
                  )}
                </div>
              )}

              {/* Cards */}
              {connectors.map((connector) => (
                <ConnectorCard
                  key={connector.id}
                  connector={connector}
                  onSettingsClick={handleOpenSettings}
                />
              ))}
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Add Connector Modal */}
      <AddConnectorModal open={addModalOpen} onOpenChange={setAddModalOpen} />

      {/* Connector Settings Modal */}
      <ConnectorSettings
        connector={settingsConnector}
        open={!!settingsConnector}
        onOpenChange={(open) => !open && setSettingsConnector(null)}
      />

      {/* Upgrade Prompt */}
      <UpgradePrompt
        open={upgradeOpen}
        onOpenChange={setUpgradeOpen}
        reason={getUpgradeReason('Unlimited data sources')}
      />
    </div>
  );
}

export default ConnectorsPage;
