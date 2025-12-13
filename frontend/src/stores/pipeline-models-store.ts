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

// Default model selections - Llama 3.3 70B for reasoning, efficient models for other tasks
const getDefaultTaskModels = (): Record<PipelineTask, TaskModelSelection> => ({
  intent: {
    provider: 'nvidia',
    modelId: 'nvidia-qwen2.5-7b',
  },
  extraction: {
    provider: 'nvidia',
    modelId: 'nvidia-mistral-nemotron',
  },
  reasoning: {
    provider: 'openrouter',
    modelId: 'openrouter-llama-3.3-70b', // NEW: Llama 3.3 70B for superior verification reasoning
  },
  summary: {
    provider: 'nvidia',
    modelId: 'nvidia-qwen2.5-7b',
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
