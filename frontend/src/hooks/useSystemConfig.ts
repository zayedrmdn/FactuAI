// Full Path: frontend/src/hooks/useSystemConfig.ts
/**
 * System Configuration Hook
 *
 * Fetches backend configuration on app startup to dynamically configure UI.
 * This is the "Single Source of Truth" bridge between backend settings and frontend.
 */

import { useState, useEffect } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface ModelsConfig {
  default_reasoning: string;
  default_intent: string;
  provider: string;
  api_base_url: string;
}

interface FeaturesConfig {
  tavily_enabled: boolean;
  learning_enabled: boolean;
  rate_limit_enabled: boolean;
  preflight_checks_enabled: boolean;
}

export interface SystemConfig {
  models: ModelsConfig;
  features: FeaturesConfig;
}

/**
 * Hook to fetch system configuration from the backend.
 *
 * @returns System configuration, loading state, and error.
 */
export function useSystemConfig() {
  const [config, setConfig] = useState<SystemConfig | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let cancelled = false;

    const fetchConfig = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/system/config`);
        if (!res.ok) {
          throw new Error('Failed to fetch system config');
        }
        const data = await res.json();
        if (!cancelled) {
          setConfig(data);
          setIsLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err : new Error('Unknown error'));
          setIsLoading(false);
        }
      }
    };

    fetchConfig();

    return () => {
      cancelled = true;
    };
  }, []);

  return {
    config,
    isLoading,
    error,
  };
}

/**
 * Find the frontend model ID that matches a backend model string.
 *
 * The backend stores model IDs like "tngtech/deepseek-r1t2-chimera:free"
 * but the frontend uses IDs like "openrouter-deepseek-r1t2-chimera".
 * This function bridges the gap.
 */
export function findFrontendModelId(
  backendModelId: string,
  models: Array<{ id: string; modelId: string }>
): string | null {
  const match = models.find((m) => m.modelId === backendModelId);
  return match?.id ?? null;
}
