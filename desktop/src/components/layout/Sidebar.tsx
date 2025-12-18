/**
 * Sidebar - Collapsible navigation sidebar
 *
 * Phase A: Dashboard Redesign
 *
 * Features:
 * - Sources list with status indicators
 * - Compact activity stream
 * - Navigation buttons (Graph, Settings)
 * - Glass/frosted styling
 * - Collapsible with smooth animation
 */

import { useNavigate } from 'react-router-dom';
import {
  FolderOpen,
  Mail,
  Github,
  Network,
  Settings,
  Clock,
  ChevronLeft,
  ChevronRight,
  Plus,
  Loader2,
  CheckCircle2,
  AlertCircle,
  PauseCircle,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useUIStore } from '@/stores/uiStore';
import { useConnectors } from '@/hooks/useApi';
import { ActivityStreamWidget } from '@/components/activity';

interface SidebarProps {
  className?: string;
}

/**
 * Get icon for connector type
 */
function getConnectorIcon(type: string | undefined) {
  switch ((type || '').toLowerCase()) {
    case 'obsidian':
      return FolderOpen;
    case 'imap':
    case 'email':
      return Mail;
    case 'github':
      return Github;
    default:
      return FolderOpen;
  }
}

/**
 * Get status indicator for connector
 */
function getStatusIndicator(status: string | undefined) {
  switch ((status || '').toLowerCase()) {
    case 'active':
    case 'syncing':
      return {
        icon: Loader2,
        className: 'text-white/60 animate-spin',
        label: 'Syncing',
      };
    case 'ready':
    case 'connected':
      return {
        icon: CheckCircle2,
        className: 'text-emerald-400/80',
        label: 'Connected',
      };
    case 'paused':
      return {
        icon: PauseCircle,
        className: 'text-amber-400/80',
        label: 'Paused',
      };
    case 'error':
      return {
        icon: AlertCircle,
        className: 'text-red-400/80',
        label: 'Error',
      };
    default:
      return {
        icon: CheckCircle2,
        className: 'text-white/40',
        label: status,
      };
  }
}

export function Sidebar({ className }: SidebarProps) {
  const navigate = useNavigate();
  const isCollapsed = useUIStore((state) => state.isSidebarCollapsed);
  const toggleSidebar = useUIStore((state) => state.toggleSidebar);
  const { data: connectors, isLoading: isLoadingConnectors } = useConnectors();

  return (
    <aside
      className={cn(
        'flex flex-col h-full border-r border-white/10',
        'bg-black/40 backdrop-blur-xl',
        'transition-all duration-300 ease-in-out',
        isCollapsed ? 'w-14' : 'w-64',
        className
      )}
    >
      {/* Sources Section */}
      <div className="flex-shrink-0">
        <div className="flex items-center justify-between px-3 py-3 border-b border-white/5">
          {!isCollapsed && (
            <span className="text-xs font-medium text-white/50 uppercase tracking-wider">
              Sources
            </span>
          )}
          {!isCollapsed && (
            <button
              onClick={() => navigate('/connectors')}
              className="p-1 text-white/40 hover:text-white/70 hover:bg-white/5 rounded transition-colors"
              title="Add source"
            >
              <Plus className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Sources List */}
        <div className="py-1">
          {isLoadingConnectors && (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="w-4 h-4 text-white/40 animate-spin" />
            </div>
          )}

          {!isLoadingConnectors && (!connectors || connectors.length === 0) && (
            <div className={cn('py-3', isCollapsed ? 'px-2' : 'px-3')}>
              {!isCollapsed && (
                <p className="text-xs text-white/30">No sources connected</p>
              )}
              <button
                onClick={() => navigate('/connectors')}
                className={cn(
                  'flex items-center gap-2 w-full mt-2 text-xs text-white/50 hover:text-white/70 transition-colors',
                  isCollapsed ? 'justify-center p-2' : 'px-2 py-1.5'
                )}
              >
                <Plus className="w-3.5 h-3.5" />
                {!isCollapsed && <span>Add your first source</span>}
              </button>
            </div>
          )}

          {connectors?.map((connector) => {
            const Icon = getConnectorIcon(connector.type);
            const status = getStatusIndicator(connector.status);
            const StatusIcon = status.icon;

            return (
              <button
                key={connector.id}
                onClick={() => navigate('/connectors')}
                className={cn(
                  'flex items-center w-full transition-colors hover:bg-white/5',
                  isCollapsed ? 'justify-center px-2 py-2.5' : 'gap-2.5 px-3 py-2'
                )}
                title={isCollapsed ? `${connector.name} (${status.label})` : undefined}
              >
                <div className="relative flex-shrink-0">
                  <Icon className="w-4 h-4 text-white/60" />
                  <StatusIcon
                    className={cn(
                      'absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5',
                      status.className
                    )}
                  />
                </div>
                {!isCollapsed && (
                  <>
                    <span className="text-sm text-white/80 truncate flex-1 text-left">
                      {connector.name}
                    </span>
                    <span className="text-[10px] text-white/40">
                      {connector.documentCount ?? 0}
                    </span>
                  </>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Activity Section */}
      <div className="flex-1 min-h-0 flex flex-col border-t border-white/5">
        <div className="flex items-center justify-between px-3 py-3 border-b border-white/5">
          {!isCollapsed && (
            <span className="text-xs font-medium text-white/50 uppercase tracking-wider">
              Activity
            </span>
          )}
          {isCollapsed && (
            <Clock className="w-4 h-4 text-white/40 mx-auto" />
          )}
          {!isCollapsed && (
            <button
              onClick={() => navigate('/activity')}
              className="text-[10px] text-white/40 hover:text-white/60 transition-colors"
            >
              View all
            </button>
          )}
        </div>

        {!isCollapsed && (
          <div className="flex-1 overflow-y-auto">
            <ActivityStreamWidget maxEvents={5} className="text-xs" />
          </div>
        )}
      </div>

      {/* Bottom Actions */}
      <div className="flex-shrink-0 border-t border-white/10 py-2">
        <div className={cn('flex', isCollapsed ? 'flex-col items-center gap-1 px-2' : 'gap-1 px-2')}>
          <button
            onClick={() => navigate('/graph')}
            className={cn(
              'flex items-center gap-2 transition-colors hover:bg-white/5 rounded',
              isCollapsed ? 'p-2.5' : 'flex-1 px-3 py-2'
            )}
            title={isCollapsed ? 'Knowledge Graph' : undefined}
          >
            <Network className="w-4 h-4 text-white/60" />
            {!isCollapsed && (
              <span className="text-xs text-white/70">Graph</span>
            )}
          </button>

          <button
            onClick={() => navigate('/settings')}
            className={cn(
              'flex items-center gap-2 transition-colors hover:bg-white/5 rounded',
              isCollapsed ? 'p-2.5' : 'flex-1 px-3 py-2'
            )}
            title={isCollapsed ? 'Settings' : undefined}
          >
            <Settings className="w-4 h-4 text-white/60" />
            {!isCollapsed && (
              <span className="text-xs text-white/70">Settings</span>
            )}
          </button>
        </div>

        {/* Collapse Toggle */}
        <div className="px-2 mt-2">
          <button
            onClick={toggleSidebar}
            className={cn(
              'flex items-center gap-2 w-full transition-colors hover:bg-white/5 rounded',
              isCollapsed ? 'justify-center p-2.5' : 'px-3 py-2'
            )}
            title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {isCollapsed ? (
              <ChevronRight className="w-4 h-4 text-white/40" />
            ) : (
              <>
                <ChevronLeft className="w-4 h-4 text-white/40" />
                <span className="text-xs text-white/40">Hide</span>
              </>
            )}
          </button>
        </div>
      </div>
    </aside>
  );
}

export default Sidebar;
