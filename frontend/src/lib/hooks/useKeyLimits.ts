/**
 * useKeyLimits Hook
 * 
 * Real-time polling of OpenRouter API key limits with safe rate limiting.
 * - Polls every 30 seconds
 * - 10-second server-side cache prevents abuse
 * - Auto-cleanup on unmount
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type { KeyLimitsResponse, KeyLimitsError } from '@/types/limits';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';
const POLL_INTERVAL = 30000; // 30 seconds

interface UseKeyLimitsResult {
  data: KeyLimitsResponse | null;
  error: string | null;
  loading: boolean;
  refetch: () => Promise<void>;
}

export function useKeyLimits(): UseKeyLimitsResult {
  const [data, setData] = useState<KeyLimitsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  const fetchLimits = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/limits`);
      
      if (!response.ok) {
        const errorData: KeyLimitsError = await response.json();
        throw new Error(errorData.error || 'Failed to fetch limits');
      }

      const limitsData: KeyLimitsResponse = await response.json();
      
      if (mountedRef.current) {
        setData(limitsData);
        setError(null);
      }
    } catch (err) {
      if (mountedRef.current) {
        setError(err instanceof Error ? err.message : 'Unknown error');
        console.error('[useKeyLimits] Fetch error:', err);
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchLimits();
  }, [fetchLimits]);

  // Setup polling
  useEffect(() => {
    intervalRef.current = setInterval(fetchLimits, POLL_INTERVAL);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [fetchLimits]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
    };
  }, []);

  return {
    data,
    error,
    loading,
    refetch: fetchLimits,
  };
}
