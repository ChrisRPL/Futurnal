/**
 * Appearance Section
 *
 * Theme, font size, and animation settings.
 * Part of the Settings page - Sovereignty Control Center.
 */

import { Palette, Type, Sparkles } from 'lucide-react';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Switch } from '@/components/ui/switch';
import { useSettingsStore } from '@/stores/settingsStore';

export function AppearanceSection() {
  const { fontSize, graphAnimations, setSetting } = useSettingsStore();

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-white">Appearance</h2>
        <p className="text-sm text-white/60 mt-1">
          Customize how Futurnal looks and feels.
        </p>
      </div>

      {/* Theme */}
      <div className="p-6 border border-white/10 bg-white/5">
        <div className="flex items-center gap-3 mb-4">
          <Palette className="h-5 w-5 text-white/60" />
          <h3 className="text-base font-medium text-white">Theme</h3>
        </div>
        <p className="text-sm text-white/50 mb-4">
          Futurnal is designed for dark mode. Light mode coming soon.
        </p>
        <div className="flex gap-4">
          <div className="flex-1 p-4 border-2 border-white/40 bg-black">
            <div className="text-sm font-medium text-white mb-1">Dark</div>
            <div className="text-xs text-white/50">Current theme</div>
          </div>
          <div className="flex-1 p-4 border border-white/10 bg-white/5 opacity-50 cursor-not-allowed">
            <div className="text-sm font-medium text-white/50 mb-1">Light</div>
            <div className="text-xs text-white/30">Coming soon</div>
          </div>
        </div>
      </div>

      {/* Font Size */}
      <div className="p-6 border border-white/10 bg-white/5">
        <div className="flex items-center gap-3 mb-4">
          <Type className="h-5 w-5 text-white/60" />
          <h3 className="text-base font-medium text-white">Font Size</h3>
        </div>
        <RadioGroup
          value={fontSize}
          onValueChange={(v) => setSetting('fontSize', v as 'small' | 'medium' | 'large')}
          className="flex gap-6"
        >
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="small" id="small" />
            <Label htmlFor="small" className="text-sm text-white/80 cursor-pointer">
              Small
            </Label>
          </div>
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="medium" id="medium" />
            <Label htmlFor="medium" className="text-base text-white/80 cursor-pointer">
              Medium
            </Label>
          </div>
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="large" id="large" />
            <Label htmlFor="large" className="text-lg text-white/80 cursor-pointer">
              Large
            </Label>
          </div>
        </RadioGroup>
      </div>

      {/* Animations */}
      <div className="p-6 border border-white/10 bg-white/5">
        <div className="flex items-center gap-3 mb-4">
          <Sparkles className="h-5 w-5 text-white/60" />
          <h3 className="text-base font-medium text-white">Animations</h3>
        </div>
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-white">Graph Animations</div>
            <p className="text-xs text-white/50 mt-1">
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
