/**
 * Search Feature Module
 *
 * Centralized export for all search configuration, types, and state management.
 * This is the single entry point for the search domain.
 *
 * Usage:
 * ```tsx
 * import {
 *   // Components
 *   SearchProvidersConfig,
 *   SearchLimitsConfig,
 *
 *   // Stores
 *   useSearchProvidersStore,
 *   useSearchLimitsStore,
 *
 *   // Types
 *   SearchProvider,
 *   SearchProviderStateItem,
 *   SEARCH_LIMITS,
 * } from '@/features/search';
 * ```
 */

// Types
export type {
  SearchProvider,
  SearchProviderStateItem,
  SearchProvidersState,
} from './stores/providers';

export type { SearchLimitsState } from './stores/limits';

// Constants
export { SEARCH_LIMITS } from './stores/limits';

// Stores
export { useSearchProvidersStore } from './stores/providers';
export { useSearchLimitsStore } from './stores/limits';

// Components
export { SearchProvidersConfig } from './components/SearchProvidersConfig';
export { SearchLimitsConfig } from './components/SearchLimitsConfig';
