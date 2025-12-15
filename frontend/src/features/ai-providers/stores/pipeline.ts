/**
 * Pipeline Model Configuration Store
 *
 * Manages model selection for different pipeline tasks:
 * - Intent Detection (lightweight)
 * - Claim Extraction (medium)
 * - Reasoning & Verification (heavyweight)
 *
 * Persists preferences to localStorage.
 * Colocated from src/stores/pipeline-models-store.ts as part of feature-centric architecture.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { getModelById } from '../registry';
import type { AIProvider } from '../types';

export type PipelineTask = 'intent' | 'extraction' | 'reasoning' | 'summary';

export interface TaskModelSelection {
  provider: AIProvider;
  modelId: string;
}

export interface PipelineModelsState {
  // Model selections per task
  intent: TaskModelSelection;
  extraction: TaskModelSelection;
  reasoning: TaskModelSelection;
  summary: TaskModelSelection;

  // Currently active task (for UI display)
  activeTask: PipelineTask | null;

  // Backend sync state
  backendSynced: boolean;

  // Actions
  setTaskModel: (task: PipelineTask, provider: AIProvider, modelId: string) => void;
  setActiveTask: (task: PipelineTask | null) => void;
  resetToDefaults: () => void;
  syncWithBackend: (backendDefaults: { reasoning?: string; intent?: string }) => void;
}

// Default model selections - OpenRouter models for all tasks
const getDefaultTaskModels = (): Record<PipelineTask, TaskModelSelection> => ({
  intent: {
    provider: 'openrouter',
    modelId: 'openrouter-glm-4.5-air', // Fast MoE model for intent detection
  },
  extraction: {
    provider: 'openrouter',
    modelId: 'openrouter-deepseek-r1t2-chimera', // Excellent reasoning for extraction
  },
  reasoning: {
    provider: 'openrouter',
    modelId: 'openrouter-llama-3.3-70b', // Llama 3.3 70B for superior verification reasoning
  },
  summary: {
    provider: 'openrouter',
    modelId: 'openrouter-glm-4.5-air', // Fast MoE model for summaries
  },
});

/**
 * Find frontend model ID from backend model string.
 * Backend uses "provider/model:tag" format, frontend uses "openrouter-model-name".
 */
function findFrontendModelIdByBackendId(backendModelId: string): string | null {
  // Try to match by extracting the model name
  // e.g., "tngtech/deepseek-r1t2-chimera:free" -> "openrouter-deepseek-r1t2-chimera"
  const match = backendModelId.match(/\/([^:]+)/);
  if (match) {
    const modelName = match[1];
    const frontendId = `openrouter-${modelName}`;
    const model = getModelById(frontendId);
    if (model) return frontendId;
  }
  return null;
}

export const usePipelineModelsStore = create<PipelineModelsState>()(
  persist(
    (set, get) => ({
      ...getDefaultTaskModels(),
      activeTask: null,
      backendSynced: false,

      setTaskModel: (task, provider, modelId) => {
        const model = getModelById(modelId);
        if (!model || model.provider !== provider) {
          console.error(`Invalid model selection: ${provider}/${modelId}`);
          return;
        }

        set({ [task]: { provider, modelId } });
      },

      setActiveTask: (task) => {
        set({ activeTask: task });
      },

      resetToDefaults: () => {
        set({ ...getDefaultTaskModels(), backendSynced: false });
      },

      syncWithBackend: (backendDefaults) => {
        if (get().backendSynced) return; // Only sync once per session

        const updates: Partial<PipelineModelsState> = { backendSynced: true };

        if (backendDefaults.reasoning) {
          const frontendId = findFrontendModelIdByBackendId(backendDefaults.reasoning);
          if (frontendId) {
            updates.reasoning = { provider: 'openrouter', modelId: frontendId };
          }
        }

        if (backendDefaults.intent) {
          const frontendId = findFrontendModelIdByBackendId(backendDefaults.intent);
          if (frontendId) {
            updates.intent = { provider: 'openrouter', modelId: frontendId };
            updates.summary = { provider: 'openrouter', modelId: frontendId };
          }
        }

        set(updates);
      },
    }),
    {
      name: 'pipeline-models-storage',
      storage: createJSONStorage(() => localStorage),
    }
  )
);
