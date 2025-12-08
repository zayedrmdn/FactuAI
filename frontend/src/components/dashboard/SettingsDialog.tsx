'use client';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { SunIcon, MoonIcon } from '@heroicons/react/24/outline';

interface Prefs {
  labelStyle: 'badge' | 'text';
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
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          {/* Theme toggle */}
          <div>
            <p className="font-medium mb-2">Theme</p>
            <div className="flex gap-2">
              <button
                onClick={() => toggleTheme(false)}
                className={`flex items-center gap-2 px-3 py-2 rounded border transition ${
                  isDark
                    ? 'border-gray-300 hover:bg-gray-50'
                    : 'bg-blue-50 border-blue-300 text-blue-700'
                }`}
              >
                <SunIcon className="w-4 h-4" />
                Light
              </button>
              <button
                onClick={() => toggleTheme(true)}
                className={`flex items-center gap-2 px-3 py-2 rounded border transition ${
                  isDark
                    ? 'bg-blue-50 dark:bg-blue-900 border-blue-300 text-blue-700 dark:text-blue-300'
                    : 'border-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800'
                }`}
              >
                <MoonIcon className="w-4 h-4" />
                Dark
              </button>
            </div>
          </div>

          {/* Text size */}
          <div>
            <p className="font-medium mb-2">Text size</p>
            <div className="space-y-2">
              {(['sm', 'md', 'lg'] as const).map((opt) => (
                <label key={opt} className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    checked={prefs.textSize === opt}
                    onChange={() => savePrefs({ textSize: opt })}
                    className="w-4 h-4"
                  />
                  <span className="capitalize">{getTextSizeLabel(opt)}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Label style */}
          <div>
            <p className="font-medium mb-2">Label style</p>
            <div className="space-y-2">
              {(['badge', 'text'] as const).map((opt) => (
                <label key={opt} className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    checked={prefs.labelStyle === opt}
                    onChange={() => savePrefs({ labelStyle: opt })}
                    className="w-4 h-4"
                  />
                  <span className="capitalize">{opt}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
