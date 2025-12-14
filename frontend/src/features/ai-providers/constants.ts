/**
 * AI Provider Constants
 * 
 * Shared constants including system prompts and default values.
 */

/**
 * System prompts for different model use cases
 */
export const SYSTEM_PROMPTS = {
    /** For fact-checking and verification tasks */
    FACTCHECK: `You are an expert fact-checking assistant. Analyze claims with precision, cite credible sources, and provide confidence scores. Be objective, thorough, and transparent about uncertainty.`,

    /** For research and deep analysis tasks */
    RESEARCH: `You are a deep research assistant. Provide comprehensive analysis, explore multiple perspectives, and synthesize information from various sources. Prioritize accuracy and depth.`,

    /** For general-purpose tasks */
    GENERAL: `You are a helpful, accurate, and concise AI assistant. Provide clear, well-reasoned responses and ask for clarification when needed.`,
} as const;

/**
 * Default provider configuration
 */
export const DEFAULT_PROVIDER = 'openrouter' as const;

/**
 * Default model ID
 */
export const DEFAULT_MODEL_ID = 'openrouter-llama-3.3-70b';
