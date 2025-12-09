/**
 * Search Providers Store
 *
 * Global state management for search provider toggles.
 * Persists user preferences to localStorage.
 *
 * Usage:
 * ```tsx
 * const { providers, toggleProvider, isProviderEnabled } = useSearchProvidersStore();
 * ```
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

/**
 * Search provider configuration
 */
export type SearchProvider = 'google' | 'newsapi';

export interface SearchProviderConfig {
  id: SearchProvider;
  name: string;
  description: string;
  enabled: boolean;
}

export interface SearchProvidersState {
  providers: SearchProviderConfig[];
  toggleProvider: (providerId: SearchProvider) => void;
  isProviderEnabled: (providerId: SearchProvider) => boolean;
  getEnabledProviders: () => SearchProvider[];
  canDisable: (providerId: SearchProvider) => boolean;
}

/**
 * Default provider configurations
 */
const defaultProviders: SearchProviderConfig[] = [
  {
    id: 'google',
    name: 'Google Search',
    description: 'Search using Google Custom Search API',
    enabled: true,
  },
  {
    id: 'newsapi',
    name: 'NewsAPI',
    description: 'Search news articles from various sources',
    enabled: true,
  },
];

/**
 * Search Providers Store implementation
 */
export const useSearchProvidersStore = create<SearchProvidersState>()(
  persist(
    (set, get) => ({
      providers: defaultProviders,

      toggleProvider: (providerId) => {
        const state = get();
        
        // Check if we can disable this provider (at least one must remain enabled)
        const currentProvider = state.providers.find((p) => p.id === providerId);
        if (!currentProvider) return;

        // If trying to disable, ensure at least one other is enabled
        if (currentProvider.enabled) {
          const enabledCount = state.providers.filter((p) => p.enabled).length;
          if (enabledCount <= 1) {
            // Cannot disable - it's the last enabled provider
            console.warn('Cannot disable the last enabled search provider');
            return;
          }
        }

        // Toggle the provider
        set({
          providers: state.providers.map((p) =>
            p.id === providerId ? { ...p, enabled: !p.enabled } : p
          ),
        });
      },

      isProviderEnabled: (providerId) => {
        const state = get();
        const provider = state.providers.find((p) => p.id === providerId);
        return provider?.enabled ?? false;
      },

      getEnabledProviders: () => {
        const state = get();
        return state.providers.filter((p) => p.enabled).map((p) => p.id);
      },

      canDisable: (providerId) => {
        const state = get();
        const currentProvider = state.providers.find((p) => p.id === providerId);
        if (!currentProvider || !currentProvider.enabled) return false;

        // Can disable if there's at least one other enabled provider
        const enabledCount = state.providers.filter((p) => p.enabled).length;
        return enabledCount > 1;
      },
    }),
    {
      name: 'search-providers',
      storage: createJSONStorage(() => localStorage),
      // Only persist the providers array
      partialize: (state) => ({ providers: state.providers }),
    }
  )
);
