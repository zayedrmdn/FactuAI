/**
 * Search Limits Store
 *
 * Global state management for search result limits (num_google, num_news, num_tavily).
 * Persists user preferences to localStorage.
 *
 * Usage:
 * ```tsx
 * const { numGoogle, numNews, numTavily, setNumGoogle, setNumNews, setNumTavily } = useSearchLimitsStore();
 * ```
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export interface SearchLimitsState {
  numGoogle: number;
  numNews: number;
  numTavily: number;
  setNumGoogle: (num: number) => void;
  setNumNews: (num: number) => void;
  setNumTavily: (num: number) => void;
  resetToDefaults: () => void;
}

/**
 * Default search result limits
 */
const DEFAULT_NUM_GOOGLE = 5;
const DEFAULT_NUM_NEWS = 5;
const DEFAULT_NUM_TAVILY = 5;

/**
 * Valid range for search limits (to prevent API abuse)
 */
export const SEARCH_LIMITS = {
  MIN: 1,
  MAX_GOOGLE: 10,
  MAX_NEWS: 100, // NewsAPI supports up to 100
  MAX_TAVILY: 10, // Tavily search results limit
} as const;

/**
 * Search Limits Store implementation
 */
export const useSearchLimitsStore = create<SearchLimitsState>()(
  persist(
    (set) => ({
      numGoogle: DEFAULT_NUM_GOOGLE,
      numNews: DEFAULT_NUM_NEWS,
      numTavily: DEFAULT_NUM_TAVILY,

      setNumGoogle: (num) => {
        // Clamp to valid range
        const clamped = Math.max(
          SEARCH_LIMITS.MIN,
          Math.min(SEARCH_LIMITS.MAX_GOOGLE, Math.floor(num))
        );
        set({ numGoogle: clamped });
      },

      setNumNews: (num) => {
        // Clamp to valid range
        const clamped = Math.max(
          SEARCH_LIMITS.MIN,
          Math.min(SEARCH_LIMITS.MAX_NEWS, Math.floor(num))
        );
        set({ numNews: clamped });
      },

      setNumTavily: (num) => {
        // Clamp to valid range
        const clamped = Math.max(
          SEARCH_LIMITS.MIN,
          Math.min(SEARCH_LIMITS.MAX_TAVILY, Math.floor(num))
        );
        set({ numTavily: clamped });
      },

      resetToDefaults: () => {
        set({
          numGoogle: DEFAULT_NUM_GOOGLE,
          numNews: DEFAULT_NUM_NEWS,
          numTavily: DEFAULT_NUM_TAVILY,
        });
      },
    }),
    {
      name: 'factuai-search-limits',
      storage: createJSONStorage(() => localStorage),
    }
  )
);
