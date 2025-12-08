"""
Model Tier Configuration

Defines lightweight and heavyweight models for different tasks.
This allows efficient resource allocation - fast models for simple tasks,
powerful models for complex reasoning.

Usage:
    from core.model_tiers import MODEL_TIERS
    intent_model = MODEL_TIERS["intent"]["openrouter"]
"""

# Model tier definitions
# Each tier maps provider -> model_id
# Model IDs match those defined in frontend/src/config/ai-models.ts
MODEL_TIERS = {
    "intent": {
        # Lightweight models for quick intent classification
        # OpenRouter: Fast, efficient nano model
        "openrouter": "nvidia/nemotron-nano-9b-v2:free",
        # NVIDIA: Default lightweight model
        "nvidia": "qwen/qwen2.5-7b-instruct",
    },
    "extraction": {
        # Medium models for claim extraction
        # OpenRouter: Reasoning-focused thinking model
        "openrouter": "allenai/olmo-3-32b-think:free",
        # NVIDIA: Efficient mid-tier model
        "nvidia": "mistralai/mistral-nemotron",
    },
    "reasoning": {
        # Heavyweight models for complex fact-checking and summarization
        # OpenRouter: Deep research model with advanced reasoning
        "openrouter": "alibaba/tongyi-deepresearch-30b-a3b:free",
        # NVIDIA: High-performance reasoning model
        "nvidia": "meta/llama-3.1-70b-instruct",
    },
}

# Default parameters for each tier
TIER_PARAMS = {
    "intent": {
        "max_tokens": 32,
        "temperature": 0.1,
    },
    "extraction": {
        "max_tokens": 512,
        "temperature": 0.3,
    },
    "reasoning": {
        "max_tokens": 2048,
        "temperature": 0.5,
    },
}


def get_model_for_tier(tier: str, provider: str = None) -> dict:
    """
    Get model configuration for a specific tier.
    
    Args:
        tier: One of "intent", "extraction", "reasoning"
        provider: LLM provider ("openrouter", "nvidia", "local")
                 If None, uses LLM_PROVIDER env var
    
    Returns:
        Dict with model_id and parameters
    """
    import os
    
    if tier not in MODEL_TIERS:
        raise ValueError(f"Unknown tier: {tier}. Valid: {list(MODEL_TIERS.keys())}")
    
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
    
    # For local mode, use the default local model
    if provider == "local":
        return {
            "provider": "local",
            "model_id": None,  # Uses default from env
            **TIER_PARAMS.get(tier, {})
        }
    
    if provider not in MODEL_TIERS[tier]:
        # Fallback to first available provider for this tier
        provider = list(MODEL_TIERS[tier].keys())[0]
    
    return {
        "provider": provider,
        "model_id": MODEL_TIERS[tier][provider],
        **TIER_PARAMS.get(tier, {})
    }
