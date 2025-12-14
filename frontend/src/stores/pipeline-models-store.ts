/**
 * Pipeline Model Configuration Store
 *
 * Manages model selection for different pipeline tasks:
 * - Intent Detection (lightweight)
 * - Claim Extraction (medium)
 * - Reasoning & Verification (heavyweight)
 *
 * Persists preferences to localStorage.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { getModelById } from '@/config/ai-models';
import type { AIProvider } from '@/types/ai';

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

  // Actions
  setTaskModel: (task: PipelineTask, provider: AIProvider, modelId: string) => void;
  setActiveTask: (task: PipelineTask | null) => void;
  resetToDefaults: () => void;
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

export const usePipelineModelsStore = create<PipelineModelsState>()(
  persist(
    (set) => ({
      ...getDefaultTaskModels(),
      activeTask: null,

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
        set(getDefaultTaskModels());
      },
    }),
    {
      name: 'pipeline-models-storage',
      storage: createJSONStorage(() => localStorage),
    }
  )
);
