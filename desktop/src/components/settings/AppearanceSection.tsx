/**
 * Appearance Section
 *
 * Theme, font size, and animation settings.
 * Part of the Settings page - Sovereignty Control Center.
 */

import { Palette, Type, Sparkles, Monitor, Sun, Moon } from 'lucide-react';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Switch } from '@/components/ui/switch';
import { useSettingsStore } from '@/stores/settingsStore';
import { useTheme } from '@/contexts/ThemeContext';
import { cn } from '@/lib/utils';

export function AppearanceSection() {
  const { fontSize, graphAnimations, setSetting } = useSettingsStore();
  const { theme, resolvedTheme, setTheme } = useTheme();

  const themeOptions = [
    { value: 'dark' as const, label: 'Dark', icon: Moon, description: 'Dark background' },
    { value: 'light' as const, label: 'Light', icon: Sun, description: 'Light background' },
    { value: 'system' as const, label: 'System', icon: Monitor, description: 'Match OS setting' },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-[var(--color-text-primary)]">Appearance</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mt-1">
          Customize how Futurnal looks and feels.
        </p>
      </div>

      {/* Theme */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <Palette className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">Theme</h3>
        </div>
        <p className="text-sm text-[var(--color-text-muted)] mb-4">
          Choose your preferred color scheme.
        </p>
        <div className="grid grid-cols-3 gap-4">
          {themeOptions.map((option) => {
            const Icon = option.icon;
            const isSelected = theme === option.value;
            return (
              <button
                key={option.value}
                onClick={() => setTheme(option.value)}
                className={cn(
                  'p-4 text-left transition-all border',
                  isSelected
                    ? 'border-[var(--color-border-active)] bg-[var(--color-surface-hover)]'
                    : 'border-[var(--color-border)] hover:border-[var(--color-border-hover)] bg-transparent'
                )}
              >
                <Icon
                  className={cn(
                    'h-5 w-5 mb-2',
                    isSelected
                      ? 'text-[var(--color-text-primary)]'
                      : 'text-[var(--color-text-tertiary)]'
                  )}
                />
                <div
                  className={cn(
                    'text-sm font-medium mb-1',
                    isSelected
                      ? 'text-[var(--color-text-primary)]'
                      : 'text-[var(--color-text-secondary)]'
                  )}
                >
                  {option.label}
                </div>
                <div className="text-xs text-[var(--color-text-muted)]">{option.description}</div>
              </button>
            );
          })}
        </div>
        {theme === 'system' && (
          <p className="text-xs text-[var(--color-text-muted)] mt-3">
            Currently using {resolvedTheme} mode based on your system settings.
          </p>
        )}
      </div>

      {/* Font Size */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <Type className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">Font Size</h3>
        </div>
        <RadioGroup
          value={fontSize}
          onValueChange={(v) => setSetting('fontSize', v as 'small' | 'medium' | 'large')}
          className="flex gap-6"
        >
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="small" id="small" />
            <Label htmlFor="small" className="text-sm text-[var(--color-text-secondary)] cursor-pointer">
              Small
            </Label>
          </div>
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="medium" id="medium" />
            <Label htmlFor="medium" className="text-base text-[var(--color-text-secondary)] cursor-pointer">
              Medium
            </Label>
          </div>
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="large" id="large" />
            <Label htmlFor="large" className="text-lg text-[var(--color-text-secondary)] cursor-pointer">
              Large
            </Label>
          </div>
        </RadioGroup>
      </div>

      {/* Animations */}
      <div className="p-6 border border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-3 mb-4">
          <Sparkles className="h-5 w-5 text-[var(--color-text-tertiary)]" />
          <h3 className="text-base font-medium text-[var(--color-text-primary)]">Animations</h3>
        </div>
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-[var(--color-text-primary)]">Graph Animations</div>
            <p className="text-xs text-[var(--color-text-muted)] mt-1">
              Enable breathing and transition animations in the knowledge graph
            </p>
          </div>
          <Switch
            checked={graphAnimations}
            onCheckedChange={(checked) => setSetting('graphAnimations', checked)}
          />
        </div>
      </div>
    </div>
  );
}
