/**
 * AI Provider Types
 * 
 * Central type definitions for the AI provider and model configuration system.
 * Colocated from src/types/ai.ts as part of feature-centric architecture.
 */

/**
 * Supported AI providers
 */
export type AIProvider = 'openrouter';

/**
 * Model capabilities and constraints
 */
export interface ModelCapabilities {
    /** Maximum context window size in tokens */
    contextWindow: number;
    /** Whether the model supports streaming responses */
    supportsStreaming: boolean;
    /** Whether the model supports function calling */
    supportsFunctionCalling: boolean;
    /** Whether the model supports vision/image inputs */
    supportsVision: boolean;
}

/**
 * Model configuration parameters
 */
export interface ModelConfig {
    /** Unique identifier for the model */
    id: string;
    /** Display name for UI */
    displayName: string;
    /** Provider that hosts this model */
    provider: AIProvider;
    /** Model identifier used in API calls */
    modelId: string;
    /** Brief description of model use case */
    description?: string;
    /** Default temperature (0-2, typically 0-1 for most models) */
    defaultTemperature: number;
    /** Default maximum tokens to generate */
    defaultMaxTokens: number;
    /** Default top-p sampling parameter */
    defaultTopP: number;
    /** Default system prompt */
    defaultSystemPrompt: string;
    /** Model capabilities */
    capabilities: ModelCapabilities;
    /** Pricing tier indicator (for display/sorting) */
    tier?: 'free' | 'low' | 'medium' | 'high' | 'premium';
    /** Whether this model is recommended for production use */
    isRecommended?: boolean;
}

/**
 * Provider configuration
 */
export interface ProviderConfig {
    /** Provider identifier */
    id: AIProvider;
    /** Display name */
    name: string;
    /** API base URL */
    baseUrl: string;
    /** Whether this provider requires API key authentication */
    requiresAuth: boolean;
    /** Available models for this provider */
    models: ModelConfig[];
    /** Provider-specific metadata */
    metadata?: {
        /** Provider website URL */
        websiteUrl?: string;
        /** Documentation URL */
        docsUrl?: string;
        /** Provider logo URL */
        logoUrl?: string;
    };
}

/**
 * Model registry structure
 */
export interface ModelRegistry {
    /** All available providers */
    providers: ProviderConfig[];
    /** Default provider ID */
    defaultProvider: AIProvider;
    /** Default model ID */
    defaultModelId: string;
}

/**
 * Runtime model selection state
 */
export interface AIModelSelection {
    /** Currently selected provider */
    provider: AIProvider;
    /** Currently selected model ID */
    modelId: string;
    /** Current session overrides for model parameters */
    sessionOverrides?:
    | {
        temperature?: number;
        maxTokens?: number;
        topP?: number;
        systemPrompt?: string;
    }
    | undefined;
}

/**
 * AI store interface for state management
 */
export interface AIStore {
    /** Current selection */
    selection: AIModelSelection;
    /** Set provider and reset to provider's default model */
    setProvider: (provider: AIProvider) => void;
    /** Set specific model */
    setModel: (modelId: string) => void;
    /** Update session overrides */
    updateOverrides: (overrides: Partial<AIModelSelection['sessionOverrides']>) => void;
    /** Reset session overrides to model defaults */
    resetOverrides: () => void;
    /** Get current model config */
    getCurrentModel: () => ModelConfig | null;
}
