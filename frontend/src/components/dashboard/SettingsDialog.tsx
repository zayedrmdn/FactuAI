'use client';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { SunIcon, MoonIcon } from '@heroicons/react/24/outline';
import { PipelineModelConfig } from '@/features/ai-providers';
import { SearchProvidersConfig } from '@/components/dashboard/SearchProvidersConfig';
import { SearchLimitsConfig } from '@/components/dashboard/SearchLimitsConfig';
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

/** Get text size label */
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
          {/* Appearance Section */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Appearance</h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              {/* Theme toggle */}
              <div>
                <p className="font-medium mb-2 text-sm">Theme</p>
                <div className="flex gap-2">
                  <button
                    onClick={() => toggleTheme(false)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-md border text-sm transition ${!isDark
                        ? 'bg-primary/10 border-primary text-primary'
                        : 'border-input hover:bg-accent hover:text-accent-foreground'
                      }`}
                  >
                    <SunIcon className="w-4 h-4" />
                    Light
                  </button>
                  <button
                    onClick={() => toggleTheme(true)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-md border text-sm transition ${isDark
                        ? 'bg-primary/10 border-primary text-primary'
                        : 'border-input hover:bg-accent hover:text-accent-foreground'
                      }`}
                  >
                    <MoonIcon className="w-4 h-4" />
                    Dark
                  </button>
                </div>
              </div>

              {/* Text size */}
              <div>
                <p className="font-medium mb-2 text-sm">Text size</p>
                <div className="flex gap-2">
                  {(['sm', 'md', 'lg'] as const).map((opt) => (
                    <button
                      key={opt}
                      onClick={() => savePrefs({ textSize: opt })}
                      className={`px-3 py-2 rounded-md border text-sm transition ${prefs.textSize === opt
                          ? 'bg-primary/10 border-primary text-primary'
                          : 'border-input hover:bg-accent hover:text-accent-foreground'
                        }`}
                    >
                      {getTextSizeLabel(opt)}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <Separator />

          {/* AI Configuration */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">AI Configuration</h3>
            <PipelineModelConfig compact textSize={prefs.textSize} />
          </div>

          <Separator />

          {/* Search Configuration */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Search Configuration</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <SearchProvidersConfig compact textSize={prefs.textSize} />
              <SearchLimitsConfig compact textSize={prefs.textSize} />
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
