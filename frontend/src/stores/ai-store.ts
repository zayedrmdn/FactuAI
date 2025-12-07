/**
 * AI Model Store
 *
 * Global state management for AI provider and model selection.
 * Persists user preferences to localStorage for session continuity.
 *
 * Usage:
 * ```tsx
 * const { selection, setProvider, setModel, getCurrentModel } = useAIStore();
 * ```
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { AIStore, AIModelSelection, ModelConfig } from '@/types/ai';
import { modelRegistry, getModelById, getProvider, isValidSelection } from '@/config/ai-models';

/**
 * Initial state with fallback to defaults
 */
const getInitialSelection = (): AIModelSelection => {
  if (typeof window !== 'undefined') {
    try {
      const raw = window.localStorage.getItem('ai-model-selection');
      if (raw) {
        const json = JSON.parse(raw) as unknown;
        let selection: AIModelSelection | undefined;
        if (typeof json === 'object' && json !== null) {
          const obj = json as { state?: { selection?: AIModelSelection } };
          if (obj.state?.selection) {
            selection = obj.state.selection;
          } else {
            const maybeSel = json as Partial<AIModelSelection>;
            if (typeof maybeSel.provider === 'string' && typeof maybeSel.modelId === 'string') {
              selection = {
                provider: maybeSel.provider,
                modelId: maybeSel.modelId,
                sessionOverrides: maybeSel.sessionOverrides,
              };
            }
          }
        }
        if (selection && isValidSelection(selection.provider, selection.modelId)) {
          return selection;
        }
      }
    } catch (error) {
      console.warn('Failed to load AI model selection from storage:', error);
    }
  }

  return {
    provider: modelRegistry.defaultProvider,
    modelId: modelRegistry.defaultModelId,
    sessionOverrides: undefined,
  };
};

/**
 * AI Store implementation
 */
export const useAIStore = create<AIStore>()(
  persist(
    (set, get) => ({
      selection: getInitialSelection(),

      setProvider: (provider) => {
        const providerConfig = getProvider(provider);
        if (!providerConfig) {
          console.error(`Provider ${provider} not found in registry`);
          return;
        }

        // Get the first recommended model, or the first model in the list
        const defaultModel =
          providerConfig.models.find((m) => m.isRecommended) || providerConfig.models[0];

        if (!defaultModel) {
          console.error(`No models found for provider ${provider}`);
          return;
        }

        set({
          selection: {
            provider,
            modelId: defaultModel.id,
            sessionOverrides: undefined, // Reset overrides when changing provider
          },
        });
      },

      setModel: (modelId) => {
        const model = getModelById(modelId);
        if (!model) {
          console.error(`Model ${modelId} not found in registry`);
          return;
        }

        set((state) => ({
          selection: {
            ...state.selection,
            provider: model.provider,
            modelId: model.id,
            // Keep existing overrides unless explicitly reset
          },
        }));
      },

      updateOverrides: (overrides) => {
        set((state) => ({
          selection: {
            ...state.selection,
            sessionOverrides: {
              ...state.selection.sessionOverrides,
              ...overrides,
            },
          },
        }));
      },

      resetOverrides: () => {
        set((state) => ({
          selection: {
            ...state.selection,
            sessionOverrides: undefined,
          },
        }));
      },

      getCurrentModel: () => {
        const { selection } = get();
        return getModelById(selection.modelId);
      },
    }),
    {
      name: 'ai-model-selection',
      storage: createJSONStorage(() => localStorage),
      // Only persist the selection, not the functions
      partialize: (state) => ({ selection: state.selection }),
    }
  )
);

/**
 * Hook to get current model with all parameters (including overrides)
 */
export function useCurrentModelConfig():
  | (ModelConfig & {
      temperature: number;
      maxTokens: number;
      topP: number;
      systemPrompt: string;
    })
  | null {
  const { selection, getCurrentModel } = useAIStore();
  const model = getCurrentModel();

  if (!model) return null;

  return {
    ...model,
    temperature: selection.sessionOverrides?.temperature ?? model.defaultTemperature,
    maxTokens: selection.sessionOverrides?.maxTokens ?? model.defaultMaxTokens,
    topP: selection.sessionOverrides?.topP ?? model.defaultTopP,
    systemPrompt: selection.sessionOverrides?.systemPrompt ?? model.defaultSystemPrompt,
  };
}

/**
 * Hook to check if current model supports a specific capability
 */
export function useModelCapability(capability: keyof ModelConfig['capabilities']): boolean {
  const model = useCurrentModelConfig();
  const value = model?.capabilities[capability];
  return typeof value === 'boolean' ? value : false;
}
