'use client';

import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Moon, Sun } from 'lucide-react';
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

function getTextSizeLabel(opt: 'sm' | 'md' | 'lg'): string {
  if (opt === 'sm') return 'Small';
  if (opt === 'md') return 'Medium';
  return 'Large';
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
          <DialogTitle>Settings</DialogTitle>
        </DialogHeader>

        <div className="space-y-8 py-4">
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Appearance</h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              <div className="space-y-2">
                <p className="font-medium text-sm">Theme</p>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant={!isDark ? 'secondary' : 'outline'}
                    size="sm"
                    onClick={() => toggleTheme(false)}
                    className="gap-2"
                  >
                    <Sun className="h-4 w-4" />
                    Light
                  </Button>
                  <Button
                    type="button"
                    variant={isDark ? 'secondary' : 'outline'}
                    size="sm"
                    onClick={() => toggleTheme(true)}
                    className="gap-2"
                  >
                    <Moon className="h-4 w-4" />
                    Dark
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <p className="font-medium text-sm">Text size</p>
                <div className="flex flex-wrap gap-2">
                  {(['sm', 'md', 'lg'] as const).map((opt) => (
                    <Button
                      key={opt}
                      type="button"
                      variant={prefs.textSize === opt ? 'secondary' : 'outline'}
                      size="sm"
                      onClick={() => savePrefs({ textSize: opt })}
                    >
                      {getTextSizeLabel(opt)}
                    </Button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <Separator />

          <div className="space-y-4">
            <h3 className="text-lg font-semibold">AI Configuration</h3>
            <PipelineModelConfig compact textSize={prefs.textSize} />
          </div>

          <Separator />

          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Search Configuration</h3>
            <SearchProvidersConfig compact textSize={prefs.textSize} />
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
