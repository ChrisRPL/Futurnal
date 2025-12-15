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
import { ChatInterface } from '@/components/chat';
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
 * Compact quick action button for inline row
 */
interface QuickActionProps {
  title: string;
  icon: React.ReactNode;
  primary?: boolean;
  onClick?: () => void;
}

function QuickAction({ title, icon, primary = false, onClick }: QuickActionProps) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-4 py-2.5 transition-all ${
        primary
          ? 'bg-[var(--color-inverse-bg)] text-[var(--color-inverse-text)] hover:bg-[var(--color-inverse-bg-hover)]'
          : 'bg-transparent border border-[var(--color-border)] text-[var(--color-text-primary)] hover:border-[var(--color-border-active)]'
      }`}
    >
      <div className={primary ? 'text-[var(--color-inverse-text)]' : 'text-[var(--color-text-tertiary)]'}>
        {icon}
      </div>
      <span className={`text-sm font-medium ${primary ? 'text-[var(--color-inverse-text)]' : 'text-[var(--color-text-primary)]'}`}>
        {title}
      </span>
    </button>
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
    <div className="min-h-screen bg-[var(--color-bg-primary)] flex flex-col">
      {/* Header with stats badges */}
      <header className="border-b border-[var(--color-border)] flex-shrink-0">
        <div className="flex items-center justify-between px-6 py-4 max-w-7xl mx-auto">
          {/* Logo */}
          <Link to="/dashboard" className="no-underline">
            <ThemeLogo variant="horizontal" className="h-7 w-auto" />
          </Link>

          {/* Center: Stats badges */}
          <div className="hidden lg:flex items-center gap-3">
            <StatBadge
              label="Nodes"
              value={totalNodes.toLocaleString()}
              icon={<CircleDot className="w-3.5 h-3.5" />}
            />
            <StatBadge
              label="Sources"
              value={sourcesConnected}
              icon={<Database className="w-3.5 h-3.5" />}
            />
            <StatBadge
              label="Memory"
              value={memoryUsage}
              icon={<HardDrive className="w-3.5 h-3.5" />}
            />
          </div>

          {/* Right side */}
          <div className="flex items-center gap-4">
            {/* Status */}
            {isLoadingStatus ? (
              <span className="text-xs text-[var(--color-text-muted)]">...</span>
            ) : (
              <div className="flex items-center gap-1.5">
                <span className={`w-1.5 h-1.5 rounded-full ${isActive ? 'bg-[var(--color-text-primary)]' : 'bg-[var(--color-text-faint)]'}`} />
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
              <div className="flex items-center gap-3">
                <span className="text-xs text-[var(--color-text-muted)] hidden md:block">{user.email}</span>

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
                >
                  <LogOut className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main - Chat-centric layout */}
      <main className="flex-1 flex flex-col px-6 py-6 max-w-5xl mx-auto w-full">
        {/* Greeting - compact */}
        <div className="mb-4">
          <h1 className="text-2xl font-brand tracking-wide text-[var(--color-text-primary)]">
            {getGreeting()}
          </h1>
          <p className="text-sm text-[var(--color-text-tertiary)]">
            Ask anything about your knowledge
          </p>
        </div>

        {/* Chat with Knowledge - HERO SECTION
            Step 03: Conversational AI
            Research Foundation:
            - ProPerSim (2509.21730v1): Proactive + personalized
            - Causal-Copilot (2504.13263v2): Natural language exploration
        */}
        <div className="flex-1 min-h-0 mb-6">
          <div className="border border-[var(--color-border)] h-full" style={{ minHeight: '50vh' }}>
            <ChatInterface
              sessionId="dashboard-main"
              onEntityClick={(entityId) => navigate(`/graph?select=${encodeURIComponent(entityId)}`)}
              enableVoice
              enableImages
              enableFiles
            />
          </div>
        </div>

        {/* Quick Actions - Compact inline row */}
        <div className="flex flex-wrap items-center gap-3 mb-6">
          <QuickAction
            title="Search"
            icon={<Search className="w-4 h-4" />}
            primary
            onClick={onOpenSearch}
          />
          <QuickAction
            title="Add Source"
            icon={<FolderPlus className="w-4 h-4" />}
            onClick={() => navigate('/connectors')}
          />
          <QuickAction
            title="Graph"
            icon={<Network className="w-4 h-4" />}
            onClick={() => navigate('/graph')}
          />
          <QuickAction
            title="Settings"
            icon={<Settings className="w-4 h-4" />}
            onClick={() => navigate('/settings')}
          />
        </div>

        {/* Knowledge Graph Mini-View - Secondary, collapsible */}
        <details className="group border border-[var(--color-border)]">
          <summary className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-[var(--color-surface)] transition-colors">
            <div className="flex items-center gap-2">
              <Network className="w-4 h-4 text-[var(--color-text-tertiary)]" />
              <span className="text-sm font-medium text-[var(--color-text-primary)]">Knowledge Graph</span>
              <span className="text-xs text-[var(--color-text-muted)]">
                {totalNodes} nodes · {sourcesConnected} sources
              </span>
            </div>
            <span className="text-xs text-[var(--color-text-tertiary)] group-open:hidden">Click to expand</span>
            <button
              onClick={(e) => { e.preventDefault(); navigate('/graph'); }}
              className="text-xs text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)] hidden group-open:block"
            >
              Open full view →
            </button>
          </summary>
          <div className="border-t border-[var(--color-border)]">
            <GraphMiniView className="h-64" />
          </div>
        </details>
      </main>

      {/* Footer - minimal */}
      <footer className="border-t border-[var(--color-surface)] px-6 py-4 flex-shrink-0">
        <div className="max-w-5xl mx-auto flex items-center justify-between text-xs text-[var(--color-text-faint)]">
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
