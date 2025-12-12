/**
 * Connectors Section
 *
 * Quick access to data source management.
 * Part of the Settings page - Sovereignty Control Center.
 */

import { useNavigate } from 'react-router-dom';
import { Database, ArrowRight, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useConnectors } from '@/hooks/useApi';

export function ConnectorsSection() {
  const navigate = useNavigate();
  const { data: connectors, isLoading } = useConnectors();

  const activeCount = connectors?.filter((c) => c.status === 'active').length ?? 0;
  const errorCount = connectors?.filter((c) => c.status === 'error').length ?? 0;
  const syncingCount = connectors?.filter((c) => c.status === 'syncing').length ?? 0;

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-white">Data Sources</h2>
        <p className="text-sm text-white/60 mt-1">
          Manage your connected data sources.
        </p>
      </div>

      {/* Quick Status */}
      <div className="p-6 border border-white/10 bg-white/5">
        <div className="flex items-center gap-3 mb-4">
          <Database className="h-5 w-5 text-white/60" />
          <h3 className="text-base font-medium text-white">Status Overview</h3>
        </div>

        {isLoading ? (
          <div className="flex items-center gap-2 text-white/50">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading connectors...
          </div>
        ) : !connectors?.length ? (
          <p className="text-sm text-white/50">No data sources configured yet.</p>
        ) : (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-white/60">Total Sources</span>
              <Badge variant="secondary">{connectors.length}</Badge>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm text-white/60">
                <CheckCircle className="h-4 w-4 text-green-500" />
                Active
              </div>
              <Badge variant="secondary">{activeCount}</Badge>
            </div>
            {syncingCount > 0 && (
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm text-white/60">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Syncing
                </div>
                <Badge variant="secondary">{syncingCount}</Badge>
              </div>
            )}
            {errorCount > 0 && (
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm text-white/60">
                  <AlertCircle className="h-4 w-4 text-red-500" />
                  Errors
                </div>
                <Badge variant="destructive">{errorCount}</Badge>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="p-6 border border-white/10 bg-white/5">
        <Button
          onClick={() => navigate('/connectors')}
          className="w-full justify-between"
        >
          <span>Manage Data Sources</span>
          <ArrowRight className="h-4 w-4" />
        </Button>
        <p className="text-xs text-white/40 mt-2 text-center">
          Add, configure, and sync your data sources
        </p>
      </div>
    </div>
  );
}
