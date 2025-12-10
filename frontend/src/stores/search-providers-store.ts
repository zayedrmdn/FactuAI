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
import { SEARCH_PROVIDERS, SearchProviderId } from '@/config/search-providers';

/**
 * Search provider configuration
 */
export type SearchProvider = SearchProviderId;

export interface SearchProviderStateItem {
  id: SearchProvider;
  name: string;
  description: string;
  enabled: boolean;
}

export interface SearchProvidersState {
  providers: SearchProviderStateItem[];
  toggleProvider: (providerId: SearchProvider) => void;
  isProviderEnabled: (providerId: SearchProvider) => boolean;
  getEnabledProviders: () => SearchProvider[];
  canDisable: (providerId: SearchProvider) => boolean;
}

/**
 * Default provider configurations from central config
 */
const defaultProviders: SearchProviderStateItem[] = SEARCH_PROVIDERS.map(p => ({
  id: p.id,
  name: p.name,
  description: p.description,
  enabled: p.defaultEnabled
}));

/**
 * Merge function to add new providers to existing localStorage data
 * This ensures backward compatibility when new providers are added
 */
const mergeProviders = (
  storedProviders: SearchProviderStateItem[] | undefined
): SearchProviderStateItem[] => {
  if (!storedProviders) return defaultProviders;

  // Create a map of stored providers by ID
  const storedMap = new Map(storedProviders.map((p) => [p.id, p]));

  // Merge: use stored config if exists, otherwise use default
  return defaultProviders.map((defaultProvider) => {
    const stored = storedMap.get(defaultProvider.id);
    if (stored) {
      // Update name/description in case they changed in config
      return {
        ...stored,
        name: defaultProvider.name,
        description: defaultProvider.description
      };
    }
    return defaultProvider; // Add new provider
  });
};

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
        if (!currentProvider?.enabled) return false;

        // Can disable if there's at least one other enabled provider
        const enabledCount = state.providers.filter((p) => p.enabled).length;
        return enabledCount > 1;
      },
    }),
    {
      name: 'search-providers',
      storage: createJSONStorage(() => localStorage),
      version: 2, // Increment version to force migration
      // Only persist the providers array
      partialize: (state) => ({ providers: state.providers }),
      // Merge stored data with new defaults to handle added providers
      merge: (persistedState, currentState) => {
        const stored = persistedState as { providers?: SearchProviderStateItem[] };
        return {
          ...currentState,
          providers: mergeProviders(stored?.providers),
        };
      },
      migrate: (persistedState: unknown, version: number) => {
        // Migration logic for version updates
        const stored = persistedState as { providers?: SearchProviderStateItem[] };
        let providers = stored?.providers || defaultProviders;

        // Version 0/1 to 2: Ensure Tavily is present
        if (version < 2) {
          providers = mergeProviders(providers);
        }

        return {
          ...(persistedState as object),
          providers,
        };
      },
    }
  )
);
