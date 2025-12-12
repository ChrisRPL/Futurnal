/**
 * Futurnal Desktop Shell - Root Application Component
 *
 * Clean, sophisticated design following Futurnal brand guidelines.
 * Cinzel for brand headlines, Times New Roman for taglines, black & white aesthetic.
 */

import { useState } from 'react';
import { Routes, Route, Navigate, Link, useNavigate } from 'react-router-dom';
import { useOrchestratorStatus, useConnectors } from '@/hooks/useApi';
import { useSubscription } from '@/hooks/useSubscription';
import { useAuth } from '@/contexts/AuthContext';
import { openSubscriptionPortal } from '@/lib/subscription';
import { ProtectedRoute } from '@/components/auth/ProtectedRoute';
import { CommandPalette } from '@/components/search';
import { GraphMiniView } from '@/components/graph';
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
    <div className="p-6 bg-white/5 border border-white/10">
      <div className="flex items-center gap-3 mb-3">
        <div className="text-white/40">{icon}</div>
        <span className="text-sm text-white/60">{label}</span>
      </div>
      <div className="text-3xl font-light text-white">{value}</div>
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
          ? 'bg-white text-black hover:bg-white/90'
          : 'bg-transparent border border-white/10 text-white hover:border-white/30'
      }`}
    >
      <div className={`w-10 h-10 mb-4 flex items-center justify-center ${
        primary ? 'text-black' : 'text-white/60'
      }`}>
        {icon}
      </div>
      <div className={`font-medium mb-1 ${primary ? 'text-black' : 'text-white'}`}>
        {title}
      </div>
      <div className={`text-sm ${primary ? 'text-black/60' : 'text-white/40'}`}>
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
    <div className="flex items-start gap-4 py-4 border-b border-white/5 last:border-0">
      <div className="text-white/40 mt-0.5">{icon}</div>
      <div className="flex-1">
        <p className="text-sm text-white/80">{message}</p>
        <p className="text-xs text-white/40 mt-1">{time}</p>
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
  const { user, logout } = useAuth();
  const { isPro, isLoading: isTierLoading } = useSubscription();

  const isActive = orchestratorStatus?.running ?? false;
  const totalNodes = 0;
  const sourcesConnected = connectors?.length ?? 0;
  const memoryUsage = '0 MB';

  const handleLogout = async () => {
    try {
      await logout();
    } catch (err) {
      console.error('Logout failed:', err);
    }
  };

  return (
    <div className="min-h-screen bg-black">
      {/* Header */}
      <header className="border-b border-white/10">
        <div className="flex items-center justify-between px-8 py-5 max-w-7xl mx-auto">
          {/* Logo */}
          <Link to="/dashboard" className="no-underline">
            <img
              src="/logo_text_horizon_dark.png"
              alt="Futurnal"
              className="h-8 w-auto"
            />
          </Link>

          {/* Right side */}
          <div className="flex items-center gap-6">
            {/* Status */}
            {isLoadingStatus ? (
              <span className="text-sm text-white/40">Connecting...</span>
            ) : (
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${isActive ? 'bg-white' : 'bg-white/30'}`} />
                <span className="text-sm text-white/60">
                  {isActive ? 'Active' : 'Idle'}
                </span>
              </div>
            )}

            {/* Tier Badge */}
            {user && !isTierLoading && (
              <span
                className={`text-xs px-2 py-1 border ${
                  isPro ? 'border-white/30 text-white/80' : 'border-white/10 text-white/40'
                }`}
              >
                {isPro ? 'Pro' : 'Free'}
              </span>
            )}

            {/* User */}
            {user && (
              <div className="flex items-center gap-4">
                <span className="text-sm text-white/40 hidden md:block">{user.email}</span>

                {/* Manage Subscription - only show for Pro */}
                {isPro && (
                  <button
                    onClick={openSubscriptionPortal}
                    className="text-sm text-white/40 hover:text-white transition-colors hidden sm:block"
                  >
                    Manage
                  </button>
                )}

                <button
                  onClick={handleLogout}
                  className="text-white/40 hover:text-white transition-colors"
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
            <h1 className="text-3xl md:text-4xl font-brand tracking-wide text-white mb-2">
              {getGreeting()}
            </h1>
            <p className="text-white/60 text-lg">
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
              <h2 className="text-lg font-medium text-white flex items-center gap-2">
                <Network className="w-5 h-5 text-white/60" />
                Knowledge Graph
              </h2>
              <button
                onClick={() => navigate('/graph')}
                className="text-xs text-white/50 hover:text-white transition-colors"
              >
                Open full view
              </button>
            </div>
            <GraphMiniView className="h-72" />
          </div>

          {/* Quick Actions */}
          <div>
            <h2 className="text-lg font-medium text-white mb-6">Quick Actions</h2>
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
            <h2 className="text-lg font-medium text-white mb-6">Recent Activity</h2>
            <div className="border border-white/10 p-6">
              {sourcesConnected === 0 ? (
                <div className="text-center py-8">
                  <p className="text-white/50">No activity yet</p>
                  <p className="text-white/30 text-sm mt-2">
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
      <footer className="border-t border-white/5 px-8 py-6 mt-12">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-white/30">
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
      <div className="min-h-screen flex items-center justify-center bg-black">
        <div className="animate-pulse-subtle">
          <img
            src="/logo_dark.png"
            alt="Loading"
            className="w-12 h-12"
          />
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
  const [searchOpen, setSearchOpen] = useState(false);

  return (
    <>
      {/* Global Command Palette - available on all pages */}
      <CommandPalette open={searchOpen} onOpenChange={setSearchOpen} />

      <Routes>
        <Route path="/" element={<RootRedirect />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <HomePage onOpenSearch={() => setSearchOpen(true)} />
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
