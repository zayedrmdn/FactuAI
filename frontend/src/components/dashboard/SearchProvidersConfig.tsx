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
import { Globe, Newspaper, AlertCircle } from 'lucide-react';

export function SearchProvidersConfig() {
  const { providers, toggleProvider, canDisable } = useSearchProvidersStore();

  const getProviderIcon = (providerId: string) => {
    switch (providerId) {
      case 'google':
        return <Globe className="h-4 w-4" />;
      case 'newsapi':
        return <Newspaper className="h-4 w-4" />;
      default:
        return null;
    }
  };

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
        {providers.map((provider) => {
          const isDisabled = !canDisable(provider.id) && provider.enabled;

          return (
            <div
              key={provider.id}
              className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3 rounded-lg border p-3 sm:p-4"
            >
              <div className="flex-1 space-y-1">
                <div className="flex items-center gap-2">
                  {getProviderIcon(provider.id)}
                  <Label
                    htmlFor={`provider-${provider.id}`}
                    className="text-sm sm:text-base font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                  >
                    {provider.name}
                  </Label>
                </div>
                <p className="text-xs sm:text-sm text-muted-foreground">{provider.description}</p>
                {isDisabled && (
                  <div className="flex items-center gap-1 text-xs text-amber-600 dark:text-amber-500">
                    <AlertCircle className="h-3 w-3 shrink-0" />
                    <span>At least one provider must be enabled</span>
                  </div>
                )}
              </div>
              <Switch
                id={`provider-${provider.id}`}
                checked={provider.enabled}
                onCheckedChange={() => toggleProvider(provider.id)}
                disabled={isDisabled}
                aria-label={`Toggle ${provider.name}`}
                className="self-start sm:self-auto"
              />
            </div>
          );
        })}

        <div className="text-xs text-muted-foreground rounded-lg bg-muted/50 p-2 sm:p-3">
          💡 <strong>Tip:</strong> Using both providers improves evidence coverage and
          reliability.
        </div>
      </CardContent>
    </Card>
  );
}
