/**
 * Search Limits Config Component
 *
 * Allows users to adjust the number of results fetched from Google and NewsAPI.
 * Responsive design for mobile, tablet, and desktop viewports.
 */

'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { useSearchLimitsStore, SEARCH_LIMITS } from '@/stores/search-limits-store';
import { useSearchProvidersStore } from '@/stores/search-providers-store';
import { Hash, RotateCcw, InfoIcon } from 'lucide-react';

export function SearchLimitsConfig() {
  const { numGoogle, numNews, setNumGoogle, setNumNews, resetToDefaults } =
    useSearchLimitsStore();
  const { isProviderEnabled } = useSearchProvidersStore();

  const handleGoogleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    if (!isNaN(value)) {
      setNumGoogle(value);
    }
  };

  const handleNewsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    if (!isNaN(value)) {
      setNumNews(value);
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
        {/* Google Search Limit */}
        <div
          className={`flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 rounded-lg border p-3 sm:p-4 transition-opacity ${
            !isProviderEnabled('google') ? 'opacity-50' : ''
          }`}
        >
          <div className="flex-1 space-y-1">
            <Label
              htmlFor="num-google"
              className="text-sm sm:text-base font-medium flex items-center gap-2"
            >
              Google Search
              {!isProviderEnabled('google') && (
                <span className="text-xs font-normal text-muted-foreground">(Disabled)</span>
              )}
            </Label>
            <p className="text-xs sm:text-sm text-muted-foreground">
              Number of Google results to fetch (1-{SEARCH_LIMITS.MAX_GOOGLE})
            </p>
          </div>
          <div className="flex items-center gap-2 sm:gap-3">
            <input
              id="num-google"
              type="number"
              min={SEARCH_LIMITS.MIN}
              max={SEARCH_LIMITS.MAX_GOOGLE}
              value={numGoogle}
              onChange={handleGoogleChange}
              disabled={!isProviderEnabled('google')}
              className="w-16 sm:w-20 rounded-md border border-input bg-background px-2 py-1.5 text-sm text-center transition-colors focus:border-ring focus:outline-none focus:ring-2 focus:ring-ring/20 disabled:cursor-not-allowed disabled:opacity-50"
              aria-label="Number of Google results"
            />
            <span className="text-xs text-muted-foreground hidden sm:inline">results</span>
          </div>
        </div>

        {/* NewsAPI Limit */}
        <div
          className={`flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 rounded-lg border p-3 sm:p-4 transition-opacity ${
            !isProviderEnabled('newsapi') ? 'opacity-50' : ''
          }`}
        >
          <div className="flex-1 space-y-1">
            <Label
              htmlFor="num-news"
              className="text-sm sm:text-base font-medium flex items-center gap-2"
            >
              NewsAPI
              {!isProviderEnabled('newsapi') && (
                <span className="text-xs font-normal text-muted-foreground">(Disabled)</span>
              )}
            </Label>
            <p className="text-xs sm:text-sm text-muted-foreground">
              Number of news articles to fetch (1-{SEARCH_LIMITS.MAX_NEWS})
            </p>
          </div>
          <div className="flex items-center gap-2 sm:gap-3">
            <input
              id="num-news"
              type="number"
              min={SEARCH_LIMITS.MIN}
              max={SEARCH_LIMITS.MAX_NEWS}
              value={numNews}
              onChange={handleNewsChange}
              disabled={!isProviderEnabled('newsapi')}
              className="w-16 sm:w-20 rounded-md border border-input bg-background px-2 py-1.5 text-sm text-center transition-colors focus:border-ring focus:outline-none focus:ring-2 focus:ring-ring/20 disabled:cursor-not-allowed disabled:opacity-50"
              aria-label="Number of NewsAPI results"
            />
            <span className="text-xs text-muted-foreground hidden sm:inline">articles</span>
          </div>
        </div>

        {/* Info Note */}
        <div className="flex items-start gap-2 rounded-lg bg-muted/50 p-2 sm:p-3">
          <InfoIcon className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
          <p className="text-xs text-muted-foreground">
            <strong>Tip:</strong> Start with lower values (3-5) for faster results. Increase if you
            need more comprehensive evidence coverage.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
