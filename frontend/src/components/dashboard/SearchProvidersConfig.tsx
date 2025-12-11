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
import { useSearchProvidersStore } from '@/stores/search-providers-store';
import { Globe, AlertCircle } from 'lucide-react';
import { getProviderConfig } from '@/config/search-providers';

export function SearchProvidersConfig({ compact = false, textSize = 'md' }: { compact?: boolean; textSize?: 'sm' | 'md' | 'lg' }) {
  const { providers, toggleProvider, canDisable } = useSearchProvidersStore();

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
        {providers.map((provider) => {
          const isDisabled = !canDisable(provider.id) && provider.enabled;
          const config = getProviderConfig(provider.id);
          const Icon = config?.icon || Globe;

          return (
            <div key={provider.id} className="flex items-center justify-between py-1">
              <div className="flex items-center gap-3">
                <Icon className="h-4 w-4 text-muted-foreground" />
                <Label 
                  htmlFor={`provider-${provider.id}`} 
                  className={`${labelClass} font-medium cursor-pointer`}
                >
                  {provider.name}
                </Label>
              </div>
              <div className="flex items-center gap-2">
                {isDisabled && (
                  <span className={`text-amber-500 font-medium uppercase tracking-wider ${descClass}`}>Required</span>
                )}
                <Switch
                  id={`provider-${provider.id}`}
                  checked={provider.enabled}
                  onCheckedChange={() => toggleProvider(provider.id)}
                  disabled={isDisabled}
                  className="scale-90 origin-right"
                />
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  const content = (
    <div className="space-y-3 sm:space-y-4">
      {providers.map((provider) => {
        const isDisabled = !canDisable(provider.id) && provider.enabled;
        const config = getProviderConfig(provider.id);
        const Icon = config?.icon || Globe;

        return (
          <div
            key={provider.id}
            className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3 rounded-lg border p-3 sm:p-4"
          >
            <div className="flex items-start gap-3">
              <div className="mt-1 rounded-full bg-primary/10 p-2 text-primary">
                <Icon className="h-4 w-4" />
              </div>
              <div className="space-y-1">
                <Label htmlFor={`provider-${provider.id}`} className={`${labelClass} font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70`}>
                  {provider.name}
                </Label>
                <p className={`${descClass} text-muted-foreground`}>
                  {provider.description}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {isDisabled && (
                <div className={`flex items-center gap-1 ${descClass} text-amber-500`} title="At least one provider must be enabled">
                  <AlertCircle className="h-3 w-3" />
                  <span className="hidden sm:inline">Required</span>
                </div>
              )}
              <Switch
                id={`provider-${provider.id}`}
                checked={provider.enabled}
                onCheckedChange={() => toggleProvider(provider.id)}
                disabled={isDisabled}
              />
            </div>
          </div>
        );
      })}
    </div>
  );

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Globe className="h-5 w-5 shrink-0" />
          <CardTitle className="text-base sm:text-lg">Search Providers</CardTitle>
        </div>
        <CardDescription className="text-xs sm:text-sm">
          Choose which sources to search for evidence. At least one must be enabled.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 sm:space-y-4">
        {content}
      </CardContent>
    </Card>
  );
}
