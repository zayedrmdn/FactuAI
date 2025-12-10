/**
 * Search Limits Config Component
 *
 * Allows users to adjust the number of results fetched from Google and NewsAPI.
 * Responsive design for mobile, tablet, and desktop viewports.
 */

'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { useSearchLimitsStore } from '@/stores/search-limits-store';
import { useSearchProvidersStore } from '@/stores/search-providers-store';
import { Hash, RotateCcw } from 'lucide-react';
import { SEARCH_PROVIDERS } from '@/config/search-providers';

export function SearchLimitsConfig() {
  const { numGoogle, numNews, numTavily, setNumGoogle, setNumNews, setNumTavily, resetToDefaults } =
    useSearchLimitsStore();
  const { isProviderEnabled } = useSearchProvidersStore();

  // Map provider IDs to their state setters and values
  // This is a bit of a bridge between the dynamic config and the specific store fields
  // In a full refactor, the store would also be dynamic (e.g. limits: Record<string, number>)
  const getProviderState = (id: string) => {
    switch (id) {
      case 'google': return { value: numGoogle, setter: setNumGoogle };
      case 'newsapi': return { value: numNews, setter: setNumNews };
      case 'tavily': return { value: numTavily, setter: setNumTavily };
      default: return null;
    }
  };

  const handleChange = (setter: (val: number) => void) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    if (!isNaN(value)) {
      setter(value);
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2 flex-1">
            <Hash className="h-5 w-5 shrink-0" />
            <CardTitle className="text-base sm:text-lg">Search Result Limits</CardTitle>
          </div>
          <button
            onClick={resetToDefaults}
            className="inline-flex items-center gap-1.5 rounded-md border bg-background px-2 py-1 text-xs font-medium transition-colors hover:bg-accent hover:text-accent-foreground shrink-0"
            title="Reset to defaults"
          >
            <RotateCcw className="h-3 w-3" />
            <span className="hidden sm:inline">Reset</span>
          </button>
        </div>
        <CardDescription className="text-xs sm:text-sm">
          Control how many articles to fetch from each source. Higher numbers provide more evidence
          but take longer.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {SEARCH_PROVIDERS.filter(p => p.hasLimit).map((provider) => {
          const state = getProviderState(provider.id);
          if (!state) return null; // Skip providers not yet in store (until store is fully dynamic)

          const isEnabled = isProviderEnabled(provider.id);

          return (
            <div
              key={provider.id}
              className={`flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 rounded-lg border p-3 sm:p-4 transition-opacity ${
                !isEnabled ? 'opacity-50' : ''
              }`}
            >
              <div className="flex-1 space-y-1">
                <Label
                  htmlFor={`num-${provider.id}`}
                  className="text-sm sm:text-base font-medium flex items-center gap-2"
                >
                  {provider.name}
                  {!isEnabled && (
                    <span className="text-xs font-normal text-muted-foreground">(Disabled)</span>
                  )}
                </Label>
                <p className="text-xs sm:text-sm text-muted-foreground">
                  Number of results to fetch (1-{provider.maxLimit})
                </p>
              </div>
              <div className="flex items-center gap-2 sm:gap-3">
                <input
                  id={`num-${provider.id}`}
                  type="number"
                  min={1}
                  max={provider.maxLimit}
                  value={state.value}
                  onChange={handleChange(state.setter)}
                  disabled={!isEnabled}
                  className="w-16 sm:w-20 rounded-md border border-input bg-background px-2 py-1.5 text-sm text-center transition-colors focus:border-ring focus:outline-none focus:ring-2 focus:ring-ring/20 disabled:cursor-not-allowed disabled:opacity-50"
                  aria-label={`Number of ${provider.name} results`}
                />
                <span className="text-xs text-muted-foreground hidden sm:inline">results</span>
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
