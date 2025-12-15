/**
 * AI Providers Feature Module
 *
 * Centralized export for all AI provider configuration, types, and state management.
 * This is the single entry point for the AI provider domain.
 *
 * Usage:
 * ```tsx
 * import {
 *   // Types
 *   AIProvider,
 *   ModelConfig,
 *   ProviderConfig,
 *
 *   // Registry
 *   modelRegistry,
 *   getModelById,
 *   getProvider,
 *
 *   // Stores
 *   useAIStore,
 *   usePipelineModelsStore,
 *
 *   // Constants
 *   SYSTEM_PROMPTS,
 * } from '@/features/ai-providers';
 * ```
 */

// Types
export type {
    AIProvider,
    ModelCapabilities,
    ModelConfig,
    ProviderConfig,
    ModelRegistry,
    AIModelSelection,
    AIStore,
} from './types';

// Constants
export { SYSTEM_PROMPTS, DEFAULT_PROVIDER, DEFAULT_MODEL_ID } from './constants';

// Registry
export {
    modelRegistry,
    getProvider,
    getModelById,
    getModelsByProvider,
    getDefaultModel,
    getRecommendedModels,
    isValidSelection,
} from './registry';

// Stores
export {
    useAIStore,
    useCurrentModelConfig,
    useModelCapability,
} from './stores/selection';

export {
    usePipelineModelsStore,
    type PipelineTask,
    type TaskModelSelection,
    type PipelineModelsState,
} from './stores/pipeline';

// Components
export { PipelineModelConfig } from './components/PipelineModelConfig';
export { ActiveModelDisplay, ModelSelector } from './components/ai-components';
