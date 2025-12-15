/**
 * Search Providers Config Component
 *
 * Allows users to toggle which search providers are used for evidence collection.
 * Ensures at least one provider is always enabled.
 */

'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Globe } from 'lucide-react';

// Full Path: src/features/search/components/SearchProvidersConfig.tsx
export function SearchProvidersConfig({
  compact = false,
  textSize = 'md',
}: {
  compact?: boolean;
  textSize?: 'sm' | 'md' | 'lg';
}) {
  const labelClass = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base',
  }[textSize];

  const descClass = {
    sm: 'text-[10px]',
    md: 'text-xs',
    lg: 'text-sm',
  }[textSize];

  if (compact) {
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between py-1">
          <div className="flex items-center gap-3">
            <Globe className="h-4 w-4 text-primary" />
            <Label className={`${labelClass} font-medium`}>Smart Web Search</Label>
          </div>
          <div className="flex items-center gap-2">
            <span className={`text-emerald-600 font-medium uppercase tracking-wider ${descClass}`}>
              Active
            </span>
            <Switch checked disabled className="scale-90 origin-right" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Globe className="h-5 w-5 shrink-0" />
          <CardTitle className="text-base sm:text-lg">Search Configuration</CardTitle>
        </div>
        <CardDescription className="text-xs sm:text-sm">
          The system automatically manages search providers.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3 rounded-lg border p-3 sm:p-4 bg-slate-50">
          <div className="flex items-start gap-3">
            <div className="mt-1 rounded-full bg-primary/10 p-2 text-primary">
              <Globe className="h-4 w-4" />
            </div>
            <div className="space-y-1">
              <Label className={`${labelClass} font-medium`}>Smart Web Search (Tavily)</Label>
              <p className={`${descClass} text-muted-foreground`}>
                Optimized search with strict domain filtering and AI summaries.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className={`flex items-center gap-1 ${descClass} text-emerald-600`}>
              <span className="hidden sm:inline font-medium">Active</span>
            </div>
            <Switch checked disabled />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
