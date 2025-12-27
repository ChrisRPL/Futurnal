/**
 * Notification Settings Section
 *
 * Phase 2D: Notification System - User preferences for proactive intelligence delivery.
 *
 * Features:
 * - Notification frequency control
 * - Do-not-disturb schedule
 * - Channel preferences (dashboard, desktop)
 * - Notification history view
 */

import { useEffect, useState } from 'react';
import {
  Bell,
  Clock,
  Moon,
  Volume2,
  VolumeX,
  Monitor,
  Layout,
  Trash2,
  ChevronDown,
  Check,
  Loader2,
  AlertCircle,
  RefreshCw,
} from 'lucide-react';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';
import { useNotificationStore } from '@/stores/notificationStore';

/** Frequency options */
const FREQUENCY_OPTIONS = [
  { value: 'realtime', label: 'Real-time', description: 'Immediate notifications' },
  { value: 'hourly', label: 'Hourly', description: 'Batched every hour' },
  { value: 'daily', label: 'Daily', description: 'Once per day digest' },
  { value: 'weekly', label: 'Weekly', description: 'Weekly summary' },
];

/**
 * Frequency Selector Component
 */
function FrequencySelector({
  value,
  onChange,
  disabled,
}: {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(false);

  const selected = FREQUENCY_OPTIONS.find((o) => o.value === value) || FREQUENCY_OPTIONS[2];

  return (
    <div className="relative">
      <button
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className={cn(
          'w-full flex items-center justify-between',
          'px-3 py-2 text-sm',
          'bg-[var(--color-bg-secondary)] border border-[var(--color-border)]',
          'text-[var(--color-text-primary)]',
          'hover:bg-[var(--color-surface-hover)] transition-colors',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
      >
        <span>{selected.label}</span>
        <ChevronDown className={cn('h-4 w-4 transition-transform', isOpen && 'rotate-180')} />
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 right-0 z-10 mt-1 bg-[var(--color-bg-secondary)] border border-[var(--color-border)] shadow-lg">
          {FREQUENCY_OPTIONS.map((option) => (
            <button
              key={option.value}
              onClick={() => {
                onChange(option.value);
                setIsOpen(false);
              }}
              className={cn(
                'w-full flex items-center justify-between px-3 py-2',
                'text-sm text-left hover:bg-[var(--color-surface-hover)]',
                option.value === value && 'bg-[var(--color-surface)]'
              )}
            >
              <div>
                <div className="font-medium text-[var(--color-text-primary)]">
                  {option.label}
                </div>
                <div className="text-xs text-[var(--color-text-muted)]">
                  {option.description}
                </div>
              </div>
              {option.value === value && <Check className="h-4 w-4 text-green-400" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Time Picker Component (simple)
 */
function TimePicker({
  value,
  onChange,
  disabled,
}: {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}) {
  return (
    <input
      type="time"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      className={cn(
        'px-3 py-2 text-sm',
        'bg-[var(--color-bg-secondary)] border border-[var(--color-border)]',
        'text-[var(--color-text-primary)]',
        'focus:outline-none focus:border-[var(--color-border-focus)]',
        disabled && 'opacity-50 cursor-not-allowed'
      )}
    />
  );
}

/**
 * Notification History Item
 */
function NotificationHistoryItem({
  notification,
  onMarkRead,
}: {
  notification: {
    notificationId: string;
    title: string;
    body: string;
    priority: string;
    createdAt: string;
    read: boolean;
  };
  onMarkRead: (id: string) => void;
}) {
  const priorityColors: Record<string, string> = {
    high: 'border-l-red-500',
    medium: 'border-l-amber-500',
    low: 'border-l-white/20',
  };

  return (
    <div
      onClick={() => !notification.read && onMarkRead(notification.notificationId)}
      className={cn(
        'p-3 border-l-2 cursor-pointer',
        'bg-[var(--color-bg-secondary)] hover:bg-[var(--color-surface-hover)]',
        priorityColors[notification.priority] || 'border-l-white/10',
        !notification.read && 'bg-white/[0.02]'
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-[var(--color-text-primary)] truncate">
              {notification.title}
            </span>
            {!notification.read && (
              <span className="w-2 h-2 rounded-full bg-blue-500 flex-shrink-0" />
            )}
          </div>
          <p className="text-xs text-[var(--color-text-muted)] line-clamp-2 mt-1">
            {notification.body}
          </p>
        </div>
        <span className="text-[10px] text-[var(--color-text-muted)] flex-shrink-0">
          {new Date(notification.createdAt).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </span>
      </div>
    </div>
  );
}

/**
 * Main NotificationSettings Component
 */
export function NotificationSettings() {
  const [showHistory, setShowHistory] = useState(false);

  const {
    preferences,
    notifications,
    status,
    unreadCount,
    isLoadingPreferences,
    isLoadingHistory,
    isSaving,
    error,
    fetchPreferences,
    fetchHistory,
    fetchStatus,
    setFrequency,
    setDnd,
    markNotificationRead,
    clearNotifications,
    deliverNotifications,
    clearError,
  } = useNotificationStore();

  // Load data on mount
  useEffect(() => {
    fetchPreferences();
    fetchStatus();
    fetchHistory(20);
  }, [fetchPreferences, fetchStatus, fetchHistory]);

  const handleFrequencyChange = async (frequency: string) => {
    await setFrequency(frequency);
  };

  const handleDndToggle = async (enabled: boolean) => {
    await setDnd({ enabled });
  };

  const handleDndTimeChange = async (startTime?: string, endTime?: string) => {
    await setDnd({ startTime, endTime });
  };

  const handleClearAll = async () => {
    if (confirm('Clear all notifications? This cannot be undone.')) {
      await clearNotifications();
    }
  };

  const handleDeliverNow = async () => {
    await deliverNotifications(true);
  };

  if (isLoadingPreferences && !preferences) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-6 w-6 text-[var(--color-text-muted)] animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-[var(--color-text-primary)]">
          Notifications
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] mt-1">
          Control how Futurnal delivers proactive insights to you.
        </p>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="p-3 bg-red-950/30 border border-red-800/50 text-red-400 text-sm flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-4 w-4" />
            <span>{error}</span>
          </div>
          <button onClick={clearError} className="text-red-300 hover:text-red-200">
            Dismiss
          </button>
        </div>
      )}

      {/* Notification Status */}
      {status && (
        <div className="p-4 bg-[var(--color-surface)] border border-[var(--color-border)]">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-[var(--color-text-primary)]">
                {status.pendingInsights}
              </div>
              <div className="text-xs text-[var(--color-text-muted)]">Pending</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-[var(--color-text-primary)]">
                {status.notificationsToday}
              </div>
              <div className="text-xs text-[var(--color-text-muted)]">Today</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-[var(--color-text-primary)]">
                {unreadCount}
              </div>
              <div className="text-xs text-[var(--color-text-muted)]">Unread</div>
            </div>
          </div>

          {status.dndActive && (
            <div className="mt-3 flex items-center justify-center gap-2 text-sm text-amber-400">
              <Moon className="h-4 w-4" />
              <span>Do Not Disturb is active</span>
            </div>
          )}
        </div>
      )}

      {/* Notification Frequency */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <Bell className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">
            Delivery Frequency
          </h3>
        </div>
        <p className="text-sm text-[var(--color-text-tertiary)] mb-4">
          Choose how often you receive insight notifications.
        </p>

        <FrequencySelector
          value={preferences?.frequency || 'daily'}
          onChange={handleFrequencyChange}
          disabled={isSaving}
        />

        <div className="mt-4 flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-[var(--color-text-primary)]">
              Daily Limit
            </div>
            <div className="text-xs text-[var(--color-text-muted)]">
              Maximum {preferences?.maxDailyNotifications || 10} notifications per day
            </div>
          </div>
          <Badge variant="secondary">
            {status?.notificationsToday || 0} / {preferences?.maxDailyNotifications || 10}
          </Badge>
        </div>
      </div>

      {/* Do Not Disturb */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <Moon className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">
            Do Not Disturb
          </h3>
          {preferences?.dndSchedule.isActive && (
            <Badge className="ml-auto bg-amber-500/20 text-amber-400 border-amber-500/30">
              Active
            </Badge>
          )}
        </div>
        <p className="text-sm text-[var(--color-text-tertiary)] mb-4">
          Set quiet hours when you don't want to be disturbed.
        </p>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-[var(--color-text-primary)]">
                Enable DND Schedule
              </div>
              <div className="text-xs text-[var(--color-text-muted)]">
                Pause notifications during scheduled hours
              </div>
            </div>
            <Switch
              checked={preferences?.dndSchedule.enabled ?? true}
              onCheckedChange={handleDndToggle}
              disabled={isSaving}
            />
          </div>

          {preferences?.dndSchedule.enabled && (
            <>
              <Separator />

              <div className="flex items-center gap-4">
                <div className="flex-1">
                  <div className="text-xs text-[var(--color-text-muted)] mb-1">Start</div>
                  <TimePicker
                    value={preferences.dndSchedule.startTime}
                    onChange={(time) => handleDndTimeChange(time, undefined)}
                    disabled={isSaving}
                  />
                </div>
                <div className="flex-1">
                  <div className="text-xs text-[var(--color-text-muted)] mb-1">End</div>
                  <TimePicker
                    value={preferences.dndSchedule.endTime}
                    onChange={(time) => handleDndTimeChange(undefined, time)}
                    disabled={isSaving}
                  />
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Channels */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <Volume2 className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">
            Notification Channels
          </h3>
        </div>
        <p className="text-sm text-[var(--color-text-tertiary)] mb-4">
          Choose where you receive notifications.
        </p>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Layout className="h-4 w-4 text-[var(--color-text-muted)]" />
              <div>
                <div className="text-sm font-medium text-[var(--color-text-primary)]">
                  Dashboard
                </div>
                <div className="text-xs text-[var(--color-text-muted)]">
                  Show in the Insights feed
                </div>
              </div>
            </div>
            <Switch checked={preferences?.channels.dashboardEnabled ?? true} disabled />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Monitor className="h-4 w-4 text-[var(--color-text-muted)]" />
              <div>
                <div className="text-sm font-medium text-[var(--color-text-primary)]">
                  Desktop Notifications
                </div>
                <div className="text-xs text-[var(--color-text-muted)]">
                  System notifications (high priority only)
                </div>
              </div>
            </div>
            <Switch checked={preferences?.channels.desktopEnabled ?? true} disabled />
          </div>
        </div>
      </div>

      {/* Notification History */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Clock className="h-5 w-5 text-[var(--color-text-tertiary)]" />
            <h3 className="text-base font-medium text-[var(--color-text-primary)]">
              Recent Notifications
            </h3>
            {unreadCount > 0 && (
              <Badge className="bg-blue-500/20 text-blue-400">{unreadCount} new</Badge>
            )}
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowHistory(!showHistory)}
          >
            {showHistory ? 'Hide' : 'Show'}
          </Button>
        </div>

        {showHistory && (
          <>
            {isLoadingHistory ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-5 w-5 text-[var(--color-text-muted)] animate-spin" />
              </div>
            ) : notifications.length > 0 ? (
              <div className="space-y-1 max-h-64 overflow-y-auto">
                {notifications.slice(0, 10).map((notification) => (
                  <NotificationHistoryItem
                    key={notification.notificationId}
                    notification={notification}
                    onMarkRead={markNotificationRead}
                  />
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-sm text-[var(--color-text-muted)]">
                <VolumeX className="h-8 w-8 mx-auto mb-2 opacity-50" />
                No notifications yet
              </div>
            )}

            {notifications.length > 0 && (
              <div className="flex gap-2 mt-4">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDeliverNow}
                  className="flex-1"
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Deliver Pending
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleClearAll}
                  className="text-red-400 hover:text-red-300 hover:bg-red-950/30"
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Clear All
                </Button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default NotificationSettings;
