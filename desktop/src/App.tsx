/**
 * Futurnal Desktop Shell - Root Application Component
 *
 * Clean, sophisticated design following Futurnal brand guidelines.
 * Cinzel for brand headlines, Times New Roman for taglines, black & white aesthetic.
 */

import { useEffect, useRef } from 'react';
import { Routes, Route, Navigate, Link, useNavigate } from 'react-router-dom';
import { useOrchestratorStatus, useConnectors, useGraphStats, useEnsureInfrastructure } from '@/hooks/useApi';
import { useSubscription } from '@/hooks/useSubscription';
import { useAuth } from '@/contexts/AuthContext';
import { openSubscriptionPortal } from '@/lib/subscription';
import { ProtectedRoute } from '@/components/auth/ProtectedRoute';
import { CommandPalette } from '@/components/search';
import { ChatInterface } from '@/components/chat';
import { ThemeLogo } from '@/components/ThemeLogo';
import { Sidebar } from '@/components/layout';
import { useUIStore } from '@/stores/uiStore';
import {
  CircleDot,
  HardDrive,
  Database,
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
import ActivityPage from '@/pages/Activity';
import InsightsPage from '@/pages/Insights';

/**
 * Compact stat badge for header
 */
interface StatBadgeProps {
  label: string;
  value: string | number;
  icon: React.ReactNode;
}

function StatBadge({ label, value, icon }: StatBadgeProps) {
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 border border-[var(--color-border)] bg-[var(--color-surface)]">
      <div className="text-[var(--color-text-muted)]">{icon}</div>
      <span className="text-xs text-[var(--color-text-tertiary)]">{label}</span>
      <span className="text-sm font-medium text-[var(--color-text-primary)]">{value}</span>
    </div>
  );
}

/**
 * Dashboard / Home page - Chat-first with sidebar layout
 */
function HomePage() {
  const navigate = useNavigate();
  const { data: orchestratorStatus, isLoading: isLoadingStatus } = useOrchestratorStatus();
  const { data: connectors } = useConnectors();
  const { data: graphStats } = useGraphStats();
  const { user, logout } = useAuth();
  const { isPro, isLoading: isTierLoading } = useSubscription();
  const openCommandPalette = useUIStore((state) => state.openCommandPalette);

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
    <div className="h-screen bg-[var(--color-bg-primary)] flex flex-col overflow-hidden">
      {/* Header - minimal */}
      <header className="border-b border-[var(--color-border)] flex-shrink-0">
        <div className="flex items-center justify-between px-4 py-3">
          {/* Logo */}
          <Link to="/dashboard" className="no-underline">
            <ThemeLogo variant="horizontal" className="h-6 w-auto" />
          </Link>

          {/* Center: Stats badges */}
          <div className="hidden md:flex items-center gap-2">
            <StatBadge
              label="Nodes"
              value={totalNodes.toLocaleString()}
              icon={<CircleDot className="w-3 h-3" />}
            />
            <StatBadge
              label="Sources"
              value={sourcesConnected}
              icon={<Database className="w-3 h-3" />}
            />
            <StatBadge
              label="Memory"
              value={memoryUsage}
              icon={<HardDrive className="w-3 h-3" />}
            />
          </div>

          {/* Right side */}
          <div className="flex items-center gap-3">
            {/* Status */}
            {isLoadingStatus ? (
              <span className="text-xs text-[var(--color-text-muted)]">...</span>
            ) : (
              <div className="flex items-center gap-1.5">
                <span className={`w-1.5 h-1.5 rounded-full ${isActive ? 'bg-emerald-400' : 'bg-[var(--color-text-faint)]'}`} />
                <span className="text-xs text-[var(--color-text-tertiary)]">
                  {isActive ? 'Active' : 'Idle'}
                </span>
              </div>
            )}

            {/* Tier Badge */}
            {user && !isTierLoading && (
              <span
                className={`text-xs px-2 py-0.5 border ${
                  isPro ? 'border-[var(--color-border-active)] text-[var(--color-text-secondary)]' : 'border-[var(--color-border)] text-[var(--color-text-muted)]'
                }`}
              >
                {isPro ? 'Pro' : 'Free'}
              </span>
            )}

            {/* User */}
            {user && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-[var(--color-text-muted)] hidden lg:block">{user.email}</span>

                {/* Manage Subscription - only show for Pro */}
                {isPro && (
                  <button
                    onClick={openSubscriptionPortal}
                    className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors hidden sm:block"
                  >
                    Manage
                  </button>
                )}

                <button
                  onClick={handleLogout}
                  className="text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors"
                  title="Logout"
                >
                  <LogOut className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main content with sidebar */}
      <div className="flex-1 flex min-h-0">
        {/* Sidebar */}
        <Sidebar />

        {/* Chat - Full width main area */}
        <main className="flex-1 flex flex-col min-h-0">
          {/* Chat Interface - Full height */}
          <div className="flex-1 min-h-0">
            <ChatInterface
              sessionId="dashboard-main"
              onEntityClick={(entityId) => navigate(`/graph?select=${encodeURIComponent(entityId)}`)}
              enableVoice
              enableImages
              enableFiles
            />
          </div>

          {/* Command hint */}
          <div className="flex-shrink-0 px-4 py-2 border-t border-[var(--color-border)] bg-[var(--color-bg-secondary)]">
            <div className="flex items-center justify-center gap-4 text-xs text-[var(--color-text-faint)]">
              <button
                onClick={openCommandPalette}
                className="flex items-center gap-1.5 hover:text-[var(--color-text-tertiary)] transition-colors"
              >
                <kbd className="px-1.5 py-0.5 bg-[var(--color-surface)] border border-[var(--color-border)] rounded text-[10px]">
                  ⌘K
                </kbd>
                <span>for commands</span>
              </button>
              <span className="text-[var(--color-text-faint)]">·</span>
              <span className="flex items-center gap-1.5">
                <kbd className="px-1.5 py-0.5 bg-[var(--color-surface)] border border-[var(--color-border)] rounded text-[10px]">
                  /
                </kbd>
                <span>for quick actions</span>
              </span>
            </div>
          </div>
        </main>
      </div>
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

  // Infrastructure auto-start on app mount (runs in background, doesn't block UI)
  const ensureInfrastructure = useEnsureInfrastructure();
  const hasInitialized = useRef(false);

  useEffect(() => {
    // Only run once on mount
    if (hasInitialized.current) return;
    hasInitialized.current = true;

    console.log('[App] Starting infrastructure services in background...');
    ensureInfrastructure.mutate(undefined, {
      onSuccess: (status) => {
        console.log('[App] Infrastructure ready:', status);
      },
      onError: (error) => {
        console.error('[App] Infrastructure startup failed:', error);
      },
    });
  }, []);

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
              <HomePage />
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
        <Route
          path="/activity"
          element={
            <ProtectedRoute>
              <ActivityPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/insights"
          element={
            <ProtectedRoute>
              <InsightsPage />
            </ProtectedRoute>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}

export default App;
