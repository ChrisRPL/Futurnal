/**
 * Futurnal Desktop Shell - Root Application Component
 *
 * This is the main application component that provides the layout structure
 * and routing for the desktop shell. Follows FRONTEND_DESIGN.md brand guidelines.
 */

import { Routes, Route } from 'react-router-dom';
import { useOrchestratorStatus, useConnectors } from '@/hooks/useApi';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Search,
  FolderPlus,
  Network,
  Settings,
  CircleDot,
  HardDrive,
  Database,
  FileText,
  Sparkles,
} from 'lucide-react';

/**
 * Get time-based greeting
 */
function getGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return 'Good morning';
  if (hour < 18) return 'Good afternoon';
  return 'Good evening';
}

/**
 * Stats card component - Brand-aligned stat display
 */
interface StatsCardProps {
  label: string;
  value: string | number;
  icon: React.ReactNode;
}

function StatsCard({ label, value, icon }: StatsCardProps) {
  return (
    <Card className="p-5">
      <div className="flex items-center gap-3 mb-2">
        <div className="text-text-secondary">{icon}</div>
        <span className="text-sm text-text-secondary">{label}</span>
      </div>
      <div className="text-2xl font-semibold text-text-primary">{value}</div>
    </Card>
  );
}

/**
 * Quick action card component - Following brand button guidelines
 */
interface QuickActionProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  variant?: 'primary' | 'secondary';
  onClick?: () => void;
}

function QuickAction({ title, description, icon, variant = 'secondary', onClick }: QuickActionProps) {
  const isPrimary = variant === 'primary';

  return (
    <Button
      variant={isPrimary ? 'default' : 'outline'}
      className={`h-auto p-5 flex flex-col items-start gap-3 text-left group ${
        isPrimary ? 'shadow-glow-primary' : 'hover:border-border-hover'
      }`}
      onClick={onClick}
    >
      <div className={`w-10 h-10 rounded-md flex items-center justify-center transition-colors ${
        isPrimary
          ? 'bg-white/20 text-white'
          : 'bg-background-elevated text-primary group-hover:bg-primary/10'
      }`}>
        {icon}
      </div>
      <div>
        <div className={`font-medium mb-0.5 ${isPrimary ? 'text-white' : 'text-text-primary'}`}>
          {title}
        </div>
        <div className={`text-sm ${isPrimary ? 'text-white/70' : 'text-text-secondary'}`}>
          {description}
        </div>
      </div>
    </Button>
  );
}

/**
 * Activity item component
 */
interface ActivityItemProps {
  icon: React.ReactNode;
  message: string;
  time: string;
  variant?: 'success' | 'info' | 'insight';
}

function ActivityItem({ icon, message, time, variant = 'info' }: ActivityItemProps) {
  const variantColors = {
    success: 'text-secondary',
    info: 'text-primary',
    insight: 'text-accent',
  };

  return (
    <div className="flex items-start gap-3 py-3 border-b border-border last:border-0">
      <div className={`mt-0.5 ${variantColors[variant]}`}>{icon}</div>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-text-primary">{message}</p>
        <p className="text-xs text-text-tertiary mt-0.5">{time}</p>
      </div>
    </div>
  );
}

/**
 * Home page component - The Ghost's Command Center
 * Design follows FRONTEND_DESIGN.md Section 3.2
 */
function HomePage() {
  const { data: orchestratorStatus, isLoading: isLoadingStatus } = useOrchestratorStatus();
  const { data: connectors } = useConnectors();

  const ghostActive = orchestratorStatus?.running ?? false;
  const totalNodes = 0; // Will be populated from PKG in future modules
  const sourcesConnected = connectors?.length ?? 0;
  const memoryUsage = '0 MB'; // Will be calculated from actual usage

  return (
    <div className="min-h-screen bg-background-deep">
      {/* Header / Navigation Bar */}
      <header className="border-b border-border bg-background-surface">
        <div className="flex items-center justify-between px-6 py-4 max-w-7xl mx-auto">
          {/* Logo & Wordmark */}
          <div className="flex items-center gap-3">
            <img
              src="/logo.png"
              alt="Futurnal"
              className="h-9 w-9"
              onError={(e) => {
                // Fallback if logo not found
                e.currentTarget.style.display = 'none';
              }}
            />
            <span className="text-xl font-semibold text-text-primary">Futurnal</span>
          </div>

          {/* Ghost Status Indicator */}
          <div className="flex items-center gap-2">
            {isLoadingStatus ? (
              <span className="text-sm text-text-tertiary animate-pulse">Connecting...</span>
            ) : ghostActive ? (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-secondary/10 border border-secondary/20">
                <span className="w-2 h-2 rounded-full bg-secondary animate-pulse" />
                <span className="text-sm font-medium text-secondary">Ghost Active</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-background-elevated border border-border">
                <span className="w-2 h-2 rounded-full bg-text-tertiary" />
                <span className="text-sm font-medium text-text-tertiary">Ghost Idle</span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="p-6 lg:p-8">
        <div className="max-w-7xl mx-auto space-y-8">

          {/* Dashboard Header - Per brand guidelines */}
          <div>
            <h1 className="text-2xl lg:text-3xl font-semibold text-text-primary">
              {getGreeting()}. {ghostActive ? 'Your Ghost is active.' : 'Your Ghost awaits.'}
            </h1>
            <p className="text-text-secondary mt-1">
              Your Personal Knowledge Graph command center.
            </p>
          </div>

          {/* Stats Row - Per brand guidelines Section 3.2 */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <StatsCard
              label="Total Nodes"
              value={totalNodes.toLocaleString()}
              icon={<CircleDot className="w-4 h-4" />}
            />
            <StatsCard
              label="Sources Connected"
              value={sourcesConnected}
              icon={<Database className="w-4 h-4" />}
            />
            <StatsCard
              label="Memory Usage"
              value={memoryUsage}
              icon={<HardDrive className="w-4 h-4" />}
            />
          </div>

          {/* Central Graph Placeholder - Per brand guidelines */}
          <Card className="p-6 min-h-[300px] flex flex-col items-center justify-center">
            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-xl bg-primary/10 flex items-center justify-center">
                <Network className="w-8 h-8 text-primary" />
              </div>
              <h3 className="text-lg font-medium text-text-primary mb-2">Knowledge Graph</h3>
              {sourcesConnected === 0 ? (
                <p className="text-text-secondary text-sm max-w-md">
                  Connect your first data source to begin building your Personal Knowledge Graph.
                  Your Ghost will extract entities and relationships automatically.
                </p>
              ) : (
                <p className="text-text-secondary text-sm max-w-md">
                  Graph visualization will appear here once entities are extracted.
                  Processing {sourcesConnected} source{sourcesConnected !== 1 ? 's' : ''}.
                </p>
              )}
            </div>
          </Card>

          {/* Quick Actions Grid */}
          <div>
            <h2 className="text-lg font-medium text-text-primary mb-4">Quick Actions</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <QuickAction
                title="Search"
                description="Query your knowledge"
                icon={<Search className="w-5 h-5" />}
                variant="primary"
                onClick={() => console.log('Navigate to Search')}
              />
              <QuickAction
                title="Add Source"
                description="Connect new data"
                icon={<FolderPlus className="w-5 h-5" />}
                onClick={() => console.log('Open Add Source dialog')}
              />
              <QuickAction
                title="Graph View"
                description="Explore connections"
                icon={<Network className="w-5 h-5" />}
                onClick={() => console.log('Navigate to Graph View')}
              />
              <QuickAction
                title="Settings"
                description="Configure your Ghost"
                icon={<Settings className="w-5 h-5" />}
                onClick={() => console.log('Navigate to Settings')}
              />
            </div>
          </div>

          {/* Recent Activity / Insights - Per brand guidelines */}
          <div>
            <h2 className="text-lg font-medium text-text-primary mb-4">Recent Activity</h2>
            <Card className="p-0 overflow-hidden">
              <div className="px-5 py-4">
                {sourcesConnected === 0 ? (
                  <div className="text-center py-8">
                    <Sparkles className="w-8 h-8 text-accent mx-auto mb-3 opacity-50" />
                    <p className="text-text-secondary">No activity yet</p>
                    <p className="text-text-tertiary text-sm mt-1">
                      Add a data source to see ingestion activity and emergent insights here.
                    </p>
                  </div>
                ) : (
                  <div>
                    <ActivityItem
                      icon={<FileText className="w-4 h-4" />}
                      message="System initialized and ready for ingestion"
                      time="Just now"
                      variant="success"
                    />
                    <ActivityItem
                      icon={<Database className="w-4 h-4" />}
                      message={`${sourcesConnected} data source${sourcesConnected !== 1 ? 's' : ''} connected`}
                      time="Just now"
                      variant="info"
                    />
                  </div>
                )}
              </div>
            </Card>
          </div>

          {/* Development Info (Dev Only) */}
          {import.meta.env.DEV && (
            <Card className="p-4 border-dashed">
              <h4 className="text-xs font-medium text-text-tertiary uppercase tracking-wider mb-2">
                Development Mode
              </h4>
              <div className="font-mono text-xs text-text-tertiary space-y-0.5">
                <p>Version: {import.meta.env.VITE_APP_VERSION ?? '0.1.0'}</p>
                <p>Environment: {import.meta.env.MODE}</p>
              </div>
            </Card>
          )}
        </div>
      </main>
    </div>
  );
}

/**
 * Main App component with routing
 */
function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      {/* Additional routes added in future modules:
          - /search - Search interface (Module 02)
          - /graph - Knowledge graph visualization (Module 03)
          - /sources - Data source management (Module 04)
          - /settings - Settings & preferences (Module 05)
      */}
    </Routes>
  );
}

export default App;
