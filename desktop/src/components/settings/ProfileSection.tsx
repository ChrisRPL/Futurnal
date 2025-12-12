/**
 * Profile Section
 *
 * User profile, subscription tier, and account management.
 * Part of the Settings page - Sovereignty Control Center.
 */

import { Crown, LogOut, Mail, Calendar } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { useAuth } from '@/contexts/AuthContext';
import { useSubscription } from '@/hooks/useSubscription';
import { openSubscriptionPortal } from '@/lib/subscription';
import { format } from 'date-fns';

export function ProfileSection() {
  const { user, logout } = useAuth();
  const { isPro, expiresAt, isLoading } = useSubscription();

  const displayName = user?.displayName || user?.email?.split('@')[0] || 'User';
  const initials = displayName
    .split(' ')
    .map((n) => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2);

  const handleLogout = async () => {
    try {
      await logout();
    } catch (err) {
      console.error('Logout failed:', err);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-white">Profile</h2>
        <p className="text-sm text-white/60 mt-1">
          Manage your account and subscription.
        </p>
      </div>

      {/* User Info */}
      <div className="p-6 border border-white/10 bg-white/5">
        <div className="flex items-center gap-4">
          <Avatar className="h-16 w-16">
            <AvatarImage src={user?.photoURL || undefined} />
            <AvatarFallback className="text-lg">{initials}</AvatarFallback>
          </Avatar>
          <div>
            <div className="text-lg font-medium text-white">
              {displayName}
            </div>
            <div className="flex items-center gap-1 text-sm text-white/60">
              <Mail className="h-3 w-3" />
              {user?.email}
            </div>
          </div>
        </div>
      </div>

      {/* Subscription */}
      <div className="p-6 border border-white/10 bg-white/5">
        <div className="flex items-center gap-3 mb-4">
          <Crown className="h-5 w-5 text-white/60" />
          <h3 className="text-base font-medium text-white">Subscription</h3>
        </div>
        {isLoading ? (
          <div className="animate-pulse h-20 bg-white/5" />
        ) : (
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-medium text-white">
                  {isPro ? 'Futurnal Pro' : 'The Archivist'}
                </span>
                <Badge
                  variant={isPro ? 'default' : 'secondary'}
                  className={isPro ? 'bg-white text-black' : ''}
                >
                  {isPro ? 'Pro' : 'Free'}
                </Badge>
              </div>
              {isPro && expiresAt && (
                <div className="flex items-center gap-1 text-xs text-white/50 mt-1">
                  <Calendar className="h-3 w-3" />
                  Renews {format(new Date(expiresAt), 'MMM d, yyyy')}
                </div>
              )}
              {!isPro && (
                <div className="text-xs text-white/50 mt-1">
                  3 data sources included
                </div>
              )}
            </div>
            {isPro ? (
              <Button variant="outline" onClick={openSubscriptionPortal}>
                Manage Subscription
              </Button>
            ) : (
              <Button onClick={openSubscriptionPortal}>
                Upgrade to Pro
              </Button>
            )}
          </div>
        )}
      </div>

      {/* Pro Features */}
      {!isPro && (
        <div className="p-6 border border-white/10 bg-white/5">
          <h3 className="text-base font-medium text-white mb-4">Pro Features</h3>
          <ul className="space-y-2 text-sm text-white/60">
            <li className="flex items-center gap-2">
              <span className="w-1 h-1 bg-white/40 rounded-full" />
              Unlimited data sources
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1 h-1 bg-white/40 rounded-full" />
              Encrypted cloud backup
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1 h-1 bg-white/40 rounded-full" />
              Emergent insights (coming soon)
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1 h-1 bg-white/40 rounded-full" />
              Causal exploration (coming soon)
            </li>
          </ul>
        </div>
      )}

      {/* Sign Out */}
      <div className="p-6 border border-white/10 bg-white/5">
        <Button variant="outline" onClick={handleLogout} className="w-full gap-2">
          <LogOut className="h-4 w-4" />
          Sign Out
        </Button>
      </div>
    </div>
  );
}
