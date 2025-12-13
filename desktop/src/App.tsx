/**
 * Futurnal Desktop Shell - Root Application Component
 *
 * Clean, sophisticated design following Futurnal brand guidelines.
 * Cinzel for brand headlines, Times New Roman for taglines, black & white aesthetic.
 */

import { Routes, Route, Navigate, Link, useNavigate } from 'react-router-dom';
import { useOrchestratorStatus, useConnectors, useGraphStats } from '@/hooks/useApi';
import { useSubscription } from '@/hooks/useSubscription';
import { useAuth } from '@/contexts/AuthContext';
import { openSubscriptionPortal } from '@/lib/subscription';
import { ProtectedRoute } from '@/components/auth/ProtectedRoute';
import { CommandPalette } from '@/components/search';
import { GraphMiniView } from '@/components/graph';
import { ThemeLogo } from '@/components/ThemeLogo';
import { useUIStore } from '@/stores/uiStore';
import {
  Search,
  FolderPlus,
  Network,
  Settings,
  CircleDot,
  HardDrive,
  Database,
  FileText,
  LogOut,
} from 'lucide-react';

// Pages
import Welcome from '@/pages/Welcome';
import Login from '@/pages/Login';
import Signup from '@/pages/Signup';
import ForgotPassword from '@/pages/ForgotPassword';
import GraphPage from '@/pages/Graph';
import ConnectorsPage from '@/pages/Connectors';
import SettingsPage from '@/pages/Settings';

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
 * Stats card component
 */
interface StatsCardProps {
  label: string;
  value: string | number;
  icon: React.ReactNode;
}

function StatsCard({ label, value, icon }: StatsCardProps) {
  return (
    <div className="p-6 bg-[var(--color-surface)] border border-[var(--color-border)]">
      <div className="flex items-center gap-3 mb-3">
        <div className="text-[var(--color-text-muted)]">{icon}</div>
        <span className="text-sm text-[var(--color-text-tertiary)]">{label}</span>
      </div>
      <div className="text-3xl font-light text-[var(--color-text-primary)]">{value}</div>
    </div>
  );
}

/**
 * Quick action button
 */
interface QuickActionProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  primary?: boolean;
  onClick?: () => void;
}

function QuickAction({ title, description, icon, primary = false, onClick }: QuickActionProps) {
  return (
    <button
      onClick={onClick}
      className={`p-6 text-left transition-all ${
        primary
          ? 'bg-[var(--color-inverse-bg)] text-[var(--color-inverse-text)] hover:bg-[var(--color-inverse-bg-hover)]'
          : 'bg-transparent border border-[var(--color-border)] text-[var(--color-text-primary)] hover:border-[var(--color-border-active)]'
      }`}
    >
      <div className={`w-10 h-10 mb-4 flex items-center justify-center ${
        primary ? 'text-[var(--color-inverse-text)]' : 'text-[var(--color-text-tertiary)]'
      }`}>
        {icon}
      </div>
      <div className={`font-medium mb-1 ${primary ? 'text-[var(--color-inverse-text)]' : 'text-[var(--color-text-primary)]'}`}>
        {title}
      </div>
      <div className={`text-sm ${primary ? 'opacity-60' : 'text-[var(--color-text-muted)]'}`}>
        {description}
      </div>
    </button>
  );
}

/**
 * Activity item
 */
interface ActivityItemProps {
  icon: React.ReactNode;
  message: string;
  time: string;
}

function ActivityItem({ icon, message, time }: ActivityItemProps) {
  return (
    <div className="flex items-start gap-4 py-4 border-b border-[var(--color-surface)] last:border-0">
      <div className="text-[var(--color-text-muted)] mt-0.5">{icon}</div>
      <div className="flex-1">
        <p className="text-sm text-[var(--color-text-secondary)]">{message}</p>
        <p className="text-xs text-[var(--color-text-muted)] mt-1">{time}</p>
      </div>
    </div>
  );
}

/**
 * Dashboard / Home page
 */
interface HomePageProps {
  /** Handler to open the search palette */
  onOpenSearch?: () => void;
}

function HomePage({ onOpenSearch }: HomePageProps) {
  const navigate = useNavigate();
  const { data: orchestratorStatus, isLoading: isLoadingStatus } = useOrchestratorStatus();
  const { data: connectors } = useConnectors();
  const { data: graphStats } = useGraphStats();
  const { user, logout } = useAuth();
  const { isPro, isLoading: isTierLoading } = useSubscription();

  const isActive = orchestratorStatus?.running ?? false;
  const totalNodes = graphStats?.total_nodes ?? 0;
  const sourcesConnected = connectors?.length ?? 0;

  // Calculate memory usage based on node count (rough estimate: ~1KB per node)
  const memoryBytes = totalNodes * 1024;
  const memoryUsage = memoryBytes < 1024 * 1024
    ? `${Math.round(memoryBytes / 1024)} KB`
    : `${(memoryBytes / (1024 * 1024)).toFixed(1)} MB`;

  const handleLogout = async () => {
    try {
      await logout();
    } catch (err) {
      console.error('Logout failed:', err);
    }
  };

  return (
    <div className="min-h-screen bg-[var(--color-bg-primary)]">
      {/* Header */}
      <header className="border-b border-[var(--color-border)]">
        <div className="flex items-center justify-between px-8 py-5 max-w-7xl mx-auto">
          {/* Logo */}
          <Link to="/dashboard" className="no-underline">
            <ThemeLogo variant="horizontal" className="h-8 w-auto" />
          </Link>

          {/* Right side */}
          <div className="flex items-center gap-6">
            {/* Status */}
            {isLoadingStatus ? (
              <span className="text-sm text-[var(--color-text-muted)]">Connecting...</span>
            ) : (
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${isActive ? 'bg-[var(--color-text-primary)]' : 'bg-[var(--color-text-faint)]'}`} />
                <span className="text-sm text-[var(--color-text-tertiary)]">
                  {isActive ? 'Active' : 'Idle'}
                </span>
              </div>
            )}

            {/* Tier Badge */}
            {user && !isTierLoading && (
              <span
                className={`text-xs px-2 py-1 border ${
                  isPro ? 'border-[var(--color-border-active)] text-[var(--color-text-secondary)]' : 'border-[var(--color-border)] text-[var(--color-text-muted)]'
                }`}
              >
                {isPro ? 'Pro' : 'Free'}
              </span>
            )}

            {/* User */}
            {user && (
              <div className="flex items-center gap-4">
                <span className="text-sm text-[var(--color-text-muted)] hidden md:block">{user.email}</span>

                {/* Manage Subscription - only show for Pro */}
                {isPro && (
                  <button
                    onClick={openSubscriptionPortal}
                    className="text-sm text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors hidden sm:block"
                  >
                    Manage
                  </button>
                )}

                <button
                  onClick={handleLogout}
                  className="text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="px-8 py-12">
        <div className="max-w-7xl mx-auto space-y-12">
          {/* Greeting */}
          <div>
            <h1 className="text-3xl md:text-4xl font-brand tracking-wide text-[var(--color-text-primary)] mb-2">
              {getGreeting()}
            </h1>
            <p className="text-[var(--color-text-tertiary)] text-lg">
              Your personal knowledge graph awaits.
            </p>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <StatsCard
              label="Knowledge Nodes"
              value={totalNodes.toLocaleString()}
              icon={<CircleDot className="w-5 h-5" />}
            />
            <StatsCard
              label="Connected Sources"
              value={sourcesConnected}
              icon={<Database className="w-5 h-5" />}
            />
            <StatsCard
              label="Memory"
              value={memoryUsage}
              icon={<HardDrive className="w-5 h-5" />}
            />
          </div>

          {/* Knowledge Graph Mini-View */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-medium text-[var(--color-text-primary)] flex items-center gap-2">
                <Network className="w-5 h-5 text-[var(--color-text-tertiary)]" />
                Knowledge Graph
              </h2>
              <button
                onClick={() => navigate('/graph')}
                className="text-xs text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] transition-colors"
              >
                Open full view
              </button>
            </div>
            <GraphMiniView className="h-72" />
          </div>

          {/* Quick Actions */}
          <div>
            <h2 className="text-lg font-medium text-[var(--color-text-primary)] mb-6">Quick Actions</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <QuickAction
                title="Search"
                description="Query your knowledge"
                icon={<Search className="w-6 h-6" />}
                primary
                onClick={onOpenSearch}
              />
              <QuickAction
                title="Add Source"
                description="Connect new data"
                icon={<FolderPlus className="w-6 h-6" />}
                onClick={() => navigate('/connectors')}
              />
              <QuickAction
                title="Graph View"
                description="Explore connections"
                icon={<Network className="w-6 h-6" />}
                onClick={() => navigate('/graph')}
              />
              <QuickAction
                title="Settings"
                description="Configure preferences"
                icon={<Settings className="w-6 h-6" />}
                onClick={() => navigate('/settings')}
              />
            </div>
          </div>

          {/* Recent Activity */}
          <div>
            <h2 className="text-lg font-medium text-[var(--color-text-primary)] mb-6">Recent Activity</h2>
            <div className="border border-[var(--color-border)] p-6">
              {sourcesConnected === 0 ? (
                <div className="text-center py-8">
                  <p className="text-[var(--color-text-tertiary)]">No activity yet</p>
                  <p className="text-[var(--color-text-faint)] text-sm mt-2">
                    Connect a data source to see activity and insights.
                  </p>
                </div>
              ) : (
                <div>
                  <ActivityItem
                    icon={<FileText className="w-4 h-4" />}
                    message="System initialized and ready"
                    time="Just now"
                  />
                  <ActivityItem
                    icon={<Database className="w-4 h-4" />}
                    message={`${sourcesConnected} data source${sourcesConnected !== 1 ? 's' : ''} connected`}
                    time="Just now"
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-[var(--color-surface)] px-8 py-6 mt-12">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-[var(--color-text-faint)]">
          <p>Futurnal v{import.meta.env.VITE_APP_VERSION ?? '0.1.0'}</p>
          <p className="font-tagline italic">Know Yourself More</p>
        </div>
      </footer>
    </div>
  );
}

/**
 * Root redirect - Welcome for guests, dashboard for authenticated users
 */
function RootRedirect() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[var(--color-bg-primary)]">
        <div className="animate-pulse-subtle">
          <ThemeLogo variant="small" className="w-12 h-12" />
        </div>
      </div>
    );
  }

  if (user) {
    return <Navigate to="/dashboard" replace />;
  }

  return <Welcome />;
}

/**
 * Main App with routing
 */
function App() {
  // Use UI store for command palette state so it can be closed from anywhere
  const isCommandPaletteOpen = useUIStore((state) => state.isCommandPaletteOpen);
  const openCommandPalette = useUIStore((state) => state.openCommandPalette);
  const closeCommandPalette = useUIStore((state) => state.closeCommandPalette);

  const handleOpenChange = (open: boolean) => {
    if (open) {
      openCommandPalette();
    } else {
      closeCommandPalette();
    }
  };

  return (
    <>
      {/* Global Command Palette - available on all pages */}
      <CommandPalette open={isCommandPaletteOpen} onOpenChange={handleOpenChange} />

      <Routes>
        <Route path="/" element={<RootRedirect />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <HomePage onOpenSearch={openCommandPalette} />
            </ProtectedRoute>
          }
        />
        <Route
          path="/graph"
          element={
            <ProtectedRoute>
              <GraphPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/connectors"
          element={
            <ProtectedRoute>
              <ConnectorsPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/settings"
          element={
            <ProtectedRoute>
              <SettingsPage />
            </ProtectedRoute>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}

export default App;
