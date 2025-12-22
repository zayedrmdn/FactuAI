'use client';

import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { SegmentedControl, SegmentOption } from '@/components/ui/segmented-control';
import { Moon, Sun, Palette, Cpu, Globe } from 'lucide-react';
import { PipelineModelConfig } from '@/features/ai-providers';
import { SearchProvidersConfig } from '@/features/search';
import { Separator } from '@/components/ui/primitives';

interface Prefs {
  textSize: 'sm' | 'md' | 'lg';
}

interface Props {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  prefs: Prefs;
  savePrefs: (p: Partial<Prefs>) => void;
  toggleTheme: (v: boolean) => void;
  isDark: boolean;
}

export default function SettingsDialog({
  open,
  onOpenChange,
  prefs,
  savePrefs,
  toggleTheme,
  isDark,
}: Readonly<Props>) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-2xl font-bold tracking-tight">Settings</DialogTitle>
        </DialogHeader>

        <div className="space-y-8 py-4">
          {/* Appearance Section */}
          <div className="space-y-5">
            {/* Section Header with Icon */}
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10 border border-primary/20 backdrop-blur-sm">
                <Palette className="h-4 w-4 text-primary" />
              </div>
              <div>
                <h3 className="text-base font-semibold">Appearance</h3>
                <p className="text-xs text-muted-foreground">Customize your visual experience</p>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              {/* Theme Selector */}
              <div className="space-y-3">
                <p className="font-medium text-sm text-muted-foreground">Theme</p>
                <SegmentedControl
                  value={isDark ? 'dark' : 'light'}
                  onValueChange={(v) => toggleTheme(v === 'dark')}
                  className="w-full"
                >
                  <SegmentOption
                    value="light"
                    icon={<Sun className="h-4 w-4" />}
                    label="Light"
                    className="flex-1"
                  />
                  <SegmentOption
                    value="dark"
                    icon={<Moon className="h-4 w-4" />}
                    label="Dark"
                    className="flex-1"
                  />
                </SegmentedControl>
              </div>

              {/* Text Size Selector */}
              <div className="space-y-3">
                <p className="font-medium text-sm text-muted-foreground">Text Size</p>
                <SegmentedControl
                  value={prefs.textSize}
                  onValueChange={(v) => savePrefs({ textSize: v as 'sm' | 'md' | 'lg' })}
                  className="w-full"
                >
                  <SegmentOption value="sm" label="A" className="flex-1 text-xs" />
                  <SegmentOption value="md" label="Aa" className="flex-1 text-sm" />
                  <SegmentOption value="lg" label="AAA" className="flex-1 text-base" />
                </SegmentedControl>
              </div>
            </div>
          </div>

          <Separator />

          {/* AI Configuration Section */}
          <div className="space-y-5">
            {/* Section Header with Icon */}
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-info/10 border border-info/20 backdrop-blur-sm">
                <Cpu className="h-4 w-4 text-info" />
              </div>
              <div>
                <h3 className="text-base font-semibold">AI Configuration</h3>
                <p className="text-xs text-muted-foreground">
                  Select models for each pipeline stage
                </p>
              </div>
            </div>

            <PipelineModelConfig compact textSize={prefs.textSize} />
          </div>

          <Separator />

          {/* Search Configuration Section */}
          <div className="space-y-5">
            {/* Section Header with Icon */}
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-success/10 border border-success/20 backdrop-blur-sm">
                <Globe className="h-4 w-4 text-success" />
              </div>
              <div>
                <h3 className="text-base font-semibold">Search Configuration</h3>
                <p className="text-xs text-muted-foreground">Manage evidence collection sources</p>
              </div>
            </div>

            <SearchProvidersConfig compact textSize={prefs.textSize} />
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
