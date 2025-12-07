/**
 * AI Model Registry
 *
 * Single source of truth for all AI providers and their models.
 *
 * To add a new model:
 * 1. Add the model configuration to the appropriate provider's models array
 * 2. Ensure all required fields are populated
 * 3. Set appropriate defaults for temperature, tokens, and system prompt
 *
 * To add a new provider:
 * 1. Add the provider ID to AIProvider type in src/types/ai.ts
 * 2. Create a new ProviderConfig entry in the providers array below
 * 3. Update defaultProvider if needed
 */

import type { ModelRegistry, ProviderConfig, ModelConfig } from '@/types/ai';

/**
 * Default system prompt for fact-checking models
 */
const FACTCHECK_SYSTEM_PROMPT = `You are an expert fact-checking assistant. Analyze claims with precision, cite credible sources, and provide confidence scores. Be objective, thorough, and transparent about uncertainty.`;

/**
 * Default system prompt for research/analysis models
 */
const RESEARCH_SYSTEM_PROMPT = `You are a deep research assistant. Provide comprehensive analysis, explore multiple perspectives, and synthesize information from various sources. Prioritize accuracy and depth.`;

/**
 * Default system prompt for general-purpose models
 */
const GENERAL_SYSTEM_PROMPT = `You are a helpful, accurate, and concise AI assistant. Provide clear, well-reasoned responses and ask for clarification when needed.`;

// ============================================================================
// OPENROUTER MODELS
// ============================================================================

const openRouterModels: ModelConfig[] = [
  {
    id: 'openrouter-tongyi-deepresearch-30b',
    displayName: 'Alibaba: Tongyi DeepResearch 30B A3B',
    provider: 'openrouter',
    modelId: 'alibaba/tongyi-deepresearch-30b-a3b:free',
    description: 'Deep research model with reasoning capabilities (Free tier)',
    defaultTemperature: 0.3,
    defaultMaxTokens: 8000,
    defaultTopP: 0.9,
    defaultSystemPrompt: RESEARCH_SYSTEM_PROMPT,
    capabilities: {
      contextWindow: 128000,
      supportsStreaming: true,
      supportsFunctionCalling: true,
      supportsVision: false,
    },
    tier: 'free',
    isRecommended: true,
  },
  {
    id: 'openrouter-olmo-3-32b',
    displayName: 'AllenAI: Olmo 3 32B Think',
    provider: 'openrouter',
    modelId: 'allenai/olmo-3-32b-think:free',
    description: 'Reasoning-focused model with enhanced thinking capabilities (Free tier)',
    defaultTemperature: 0.2,
    defaultMaxTokens: 6000,
    defaultTopP: 0.85,
    defaultSystemPrompt: RESEARCH_SYSTEM_PROMPT,
    capabilities: {
      contextWindow: 32768,
      supportsStreaming: true,
      supportsFunctionCalling: false,
      supportsVision: false,
    },
    tier: 'free',
    isRecommended: false,
  },
  {
    id: 'openrouter-gpt-oss-120b',
    displayName: 'OpenAI: GPT-OSS 120B',
    provider: 'openrouter',
    modelId: 'openai/gpt-oss-120b:free',
    description: 'Large open-source model for complex reasoning tasks (Free tier)',
    defaultTemperature: 0.7,
    defaultMaxTokens: 4096,
    defaultTopP: 0.9,
    defaultSystemPrompt: GENERAL_SYSTEM_PROMPT,
    capabilities: {
      contextWindow: 8192,
      supportsStreaming: true,
      supportsFunctionCalling: true,
      supportsVision: false,
    },
    tier: 'free',
    isRecommended: false,
  },
  {
    id: 'openrouter-nemotron-nano-9b',
    displayName: 'NVIDIA: Nemotron Nano 9B V2',
    provider: 'openrouter',
    modelId: 'nvidia/nemotron-nano-9b-v2:free',
    description: 'Fast, efficient model for quick fact-checking tasks (Free tier)',
    defaultTemperature: 0.5,
    defaultMaxTokens: 2048,
    defaultTopP: 0.9,
    defaultSystemPrompt: FACTCHECK_SYSTEM_PROMPT,
    capabilities: {
      contextWindow: 4096,
      supportsStreaming: true,
      supportsFunctionCalling: false,
      supportsVision: false,
    },
    tier: 'free',
    isRecommended: false,
  },
  {
    id: 'openrouter-longcat-flash',
    displayName: 'Meituan: LongCat Flash Chat',
    provider: 'openrouter',
    modelId: 'meituan/longcat-flash-chat:free',
    description: 'Ultra-fast model with extended context for conversational tasks (Free tier)',
    defaultTemperature: 0.8,
    defaultMaxTokens: 4096,
    defaultTopP: 0.95,
    defaultSystemPrompt: GENERAL_SYSTEM_PROMPT,
    capabilities: {
      contextWindow: 32768,
      supportsStreaming: true,
      supportsFunctionCalling: false,
      supportsVision: false,
    },
    tier: 'free',
    isRecommended: false,
  },
];

// ============================================================================
// NVIDIA NIM MODELS
// ============================================================================

const nvidiaModels: ModelConfig[] = [
  {
    id: 'nvidia-llama-3.1-405b',
    displayName: 'Meta Llama 3.1 405B Instruct',
    provider: 'nvidia',
    modelId: 'meta/llama-3.1-405b-instruct',
    description: 'State-of-the-art large language model for complex reasoning and analysis',
    defaultTemperature: 0.2,
    defaultMaxTokens: 1024,
    defaultTopP: 0.7,
    defaultSystemPrompt: RESEARCH_SYSTEM_PROMPT,
    capabilities: {
      contextWindow: 128000,
      supportsStreaming: true,
      supportsFunctionCalling: true,
      supportsVision: false,
    },
    tier: 'premium',
    isRecommended: true,
  },
  {
    id: 'nvidia-llama-3.1-70b',
    displayName: 'Meta Llama 3.1 70B Instruct',
    provider: 'nvidia',
    modelId: 'meta/llama-3.1-70b-instruct',
    description: 'Balanced model offering strong performance for most tasks',
    defaultTemperature: 0.2,
    defaultMaxTokens: 1024,
    defaultTopP: 0.7,
    defaultSystemPrompt: FACTCHECK_SYSTEM_PROMPT,
    capabilities: {
      contextWindow: 128000,
      supportsStreaming: true,
      supportsFunctionCalling: true,
      supportsVision: false,
    },
    tier: 'high',
    isRecommended: false,
  },
  {
    id: 'nvidia-llama-3.1-8b',
    displayName: 'Meta Llama 3.1 8B Instruct',
    provider: 'nvidia',
    modelId: 'meta/llama-3.1-8b-instruct',
    description: 'Lightweight, fast model for quick responses',
    defaultTemperature: 0.2,
    defaultMaxTokens: 1024,
    defaultTopP: 0.7,
    defaultSystemPrompt: GENERAL_SYSTEM_PROMPT,
    capabilities: {
      contextWindow: 128000,
      supportsStreaming: true,
      supportsFunctionCalling: false,
      supportsVision: false,
    },
    tier: 'low',
    isRecommended: false,
  },
  {
    id: 'nvidia-mistral-nemotron',
    displayName: 'Mistral Nemotron',
    provider: 'nvidia',
    modelId: 'mistralai/mistral-nemotron',
    description: 'Efficient model with strong reasoning capabilities',
    defaultTemperature: 0.6,
    defaultMaxTokens: 4096,
    defaultTopP: 0.7,
    defaultSystemPrompt: FACTCHECK_SYSTEM_PROMPT,
    capabilities: {
      contextWindow: 32768,
      supportsStreaming: true,
      supportsFunctionCalling: true,
      supportsVision: false,
    },
    tier: 'medium',
    isRecommended: false,
  },
  {
    id: 'nvidia-qwen2.5-7b',
    displayName: 'Qwen 2.5 7B Instruct',
    provider: 'nvidia',
    modelId: 'qwen/qwen2.5-7b-instruct',
    description: 'Fast and efficient model for general tasks (Default)',
    defaultTemperature: 0.2,
    defaultMaxTokens: 1024,
    defaultTopP: 0.7,
    defaultSystemPrompt: GENERAL_SYSTEM_PROMPT,
    capabilities: {
      contextWindow: 32768,
      supportsStreaming: true,
      supportsFunctionCalling: true,
      supportsVision: false,
    },
    tier: 'low',
    isRecommended: true,
  },
];

// ============================================================================
// PROVIDER CONFIGURATIONS
// ============================================================================

const providers: ProviderConfig[] = [
  {
    id: 'openrouter',
    name: 'OpenRouter',
    baseUrl: 'https://openrouter.ai/api/v1',
    requiresAuth: true,
    models: openRouterModels,
    metadata: {
      websiteUrl: 'https://openrouter.ai',
      docsUrl: 'https://openrouter.ai/docs',
    },
  },
  {
    id: 'nvidia',
    name: 'NVIDIA NIM',
    baseUrl: 'https://integrate.api.nvidia.com/v1',
    requiresAuth: true,
    models: nvidiaModels,
    metadata: {
      websiteUrl: 'https://www.nvidia.com/en-us/ai/',
      docsUrl: 'https://docs.api.nvidia.com/',
    },
  },
];

// ============================================================================
// REGISTRY EXPORT
// ============================================================================

/**
 * Main model registry - single source of truth
 */
export const modelRegistry: ModelRegistry = {
  providers,
  defaultProvider: 'nvidia',
  defaultModelId: 'nvidia-qwen2.5-7b', // Default: Fast and efficient
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get a provider by ID
 */
export function getProvider(providerId: string): ProviderConfig | null {
  return providers.find((p) => p.id === providerId) || null;
}

/**
 * Get a model by ID across all providers
 */
export function getModelById(modelId: string): ModelConfig | null {
  for (const provider of providers) {
    const model = provider.models.find((m) => m.id === modelId);
    if (model) return model;
  }
  return null;
}

/**
 * Get all models for a specific provider
 */
export function getModelsByProvider(providerId: string): ModelConfig[] {
  const provider = getProvider(providerId);
  return provider?.models || [];
}

/**
 * Get the default model configuration
 */
export function getDefaultModel(): ModelConfig {
  const model = getModelById(modelRegistry.defaultModelId);
  if (!model) {
    throw new Error(`Default model ${modelRegistry.defaultModelId} not found in registry`);
  }
  return model;
}

/**
 * Get recommended models across all providers
 */
export function getRecommendedModels(): ModelConfig[] {
  return providers.flatMap((p) => p.models).filter((m) => m.isRecommended);
}

/**
 * Validate if a provider/model combination exists
 */
export function isValidSelection(providerId: string, modelId: string): boolean {
  const model = getModelById(modelId);
  return model !== null && model.provider === providerId;
}
