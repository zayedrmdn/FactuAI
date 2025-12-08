"""
Service Manager

Manages shared service instances across the application.
Implements singleton pattern for expensive-to-load services.

Supports dual run modes:
- Cloud Mode: Uses API-based LLM and LLM-based classifier
- Local Mode: Uses local Unsloth LLM and DistilBERT classifier
"""

import os
from typing import Optional, Tuple

from core.config import Config
from core.logging import logger


class ServiceManager:
    """
    Manages shared service instances across the application.
    
    Services are initialized lazily and conditionally based on
    APP_RUN_MODE configuration.
    """
    
    _instance: Optional['ServiceManager'] = None
    _llm_client = None
    _classifier = None
    _search_client = None
    _pipeline_orchestrator = None
    _ocr_service = None
    _keybert_model = None
    _sentence_transformer = None
    _llm_cache = {}  # Cache for dynamic LLM instances
    _initialized: bool = False
    
    def __new__(cls) -> 'ServiceManager':
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize_services(self) -> None:
        """Initialize all shared services based on run mode."""
        if self._initialized:
            logger.debug("[SERVICE] Services already initialized; skipping")
            return
        
        run_mode = Config.APP_RUN_MODE
        logger.info(f"[SERVICE] Initializing services in {run_mode.upper()} mode...")
        
        try:
            # Initialize LLM client using factory
            logger.debug("[SERVICE] Loading LLM client...")
            self._init_llm_client()
            
            # Initialize classifier based on mode
            logger.debug("[SERVICE] Loading classifier...")
            self._init_classifier()
            
            # Initialize search client (same for both modes)
            logger.debug("[SERVICE] Loading search client...")
            self._init_search_client()
            
            self._initialized = True
            logger.info("[SERVICE] All services initialized successfully")
            
        except Exception as e:
            logger.error(f"[SERVICE] Failed to initialize services: {e}")
            raise
    
    def _init_llm_client(self) -> None:
        """Initialize LLM client using factory."""
        from services.llm.factory import LLMFactory
        from core.logging import log_model_init
        
        self._llm_client = LLMFactory.create()
        
        if self._llm_client.is_available():
            info = self._llm_client.get_model_info()
            provider = info.get('provider', 'unknown')
            model_name = info.get('model_name', 'default')
            log_model_init(logger, provider, model_name, "success")
            logger.info(f"[SERVICE] LLM initialized: {provider}")
        else:
            logger.warning("[SERVICE] LLM client initialized but not available")
            log_model_init(logger, "unknown", "unknown", "failed")
    
    def _init_classifier(self) -> None:
        """Initialize classifier based on run mode."""
        if Config.USE_LOCAL_CLASSIFIER:
            # Local mode: use DistilBERT classifier
            try:
                from services.classifier.client import ClaimClassifier
                self._classifier = ClaimClassifier()
                logger.info("[SERVICE] Using local DistilBERT classifier")
            except Exception as e:
                logger.error(f"[SERVICE] Failed to load local classifier: {e}")
                # Fallback to LLM classifier
                self._init_llm_classifier()
        else:
            # Cloud mode: use LLM-based classifier
            self._init_llm_classifier()
    
    def _init_llm_classifier(self) -> None:
        """Initialize LLM-based classifier."""
        from services.classifier.llm_classifier import LLMClassifier
        self._classifier = LLMClassifier(llm=self._llm_client)
        logger.info("[SERVICE] Using LLM-based classifier")
    
    def _init_search_client(self) -> None:
        """Initialize search client."""
        from services.search.google_search import GoogleSearchClient
        self._search_client = GoogleSearchClient()
    
    def get_llm_client(self):
        """Get the shared LLM client instance."""
        if not self._initialized:
            self.initialize_services()
        
        if self._llm_client is None:
            raise RuntimeError("[SERVICE] LLM client not available")
        
        return self._llm_client
    
    def get_classifier(self):
        """Get the shared classifier instance."""
        if not self._initialized:
            self.initialize_services()
        
        if self._classifier is None:
            raise RuntimeError("[SERVICE] Classifier not available")
        
        return self._classifier
    
    def get_search_client(self):
        """Get the shared search client instance."""
        if not self._initialized:
            self.initialize_services()
        
        if self._search_client is None:
            raise RuntimeError("[SERVICE] Search client not available")
        
        return self._search_client
    
    def get_all_services(self) -> Tuple:
        """Get all service instances."""
        return (
            self.get_llm_client(),
            self.get_classifier(),
            self.get_search_client()
        )
    
    def get_pipeline_orchestrator(self):
        """Get the shared pipeline orchestrator instance (singleton)."""
        if self._pipeline_orchestrator is None:
            from services.factcheck_service import PipelineOrchestrator
            logger.debug("[SERVICE] Creating singleton PipelineOrchestrator")
            self._pipeline_orchestrator = PipelineOrchestrator()
        return self._pipeline_orchestrator
    
    def get_ocr_service(self):
        """Get the shared OCR service instance (singleton)."""
        if self._ocr_service is None:
            from services.ocr import OCRService
            logger.debug("[SERVICE] Creating singleton OCRService")
            self._ocr_service = OCRService()
        return self._ocr_service
    
    def get_keybert_model(self):
        """Get the shared KeyBERT model instance (singleton, expensive to load)."""
        if self._keybert_model is None:
            try:
                from keybert import KeyBERT
                logger.info("[SERVICE] Loading KeyBERT model (one-time initialization)...")
                self._keybert_model = KeyBERT()
                logger.info("[SERVICE] KeyBERT model loaded successfully")
            except Exception as e:
                logger.error(f"[SERVICE] Failed to load KeyBERT: {e}")
                raise
        return self._keybert_model
    
    def get_sentence_transformer(self, model_name: str = "all-MiniLM-L6-v2"):
        """Get the shared SentenceTransformer instance (singleton, expensive to load)."""
        if self._sentence_transformer is None:
            try:
                from sentence_transformers import SentenceTransformer
                import torch
                logger.info(f"[SERVICE] Loading SentenceTransformer ({model_name})...")
                self._sentence_transformer = SentenceTransformer(model_name)
                # Move to GPU if available
                if torch.cuda.is_available():
                    self._sentence_transformer = self._sentence_transformer.to("cuda")
                    logger.info("[SERVICE] SentenceTransformer loaded on GPU")
                else:
                    logger.info("[SERVICE] SentenceTransformer loaded on CPU")
            except Exception as e:
                logger.error(f"[SERVICE] Failed to load SentenceTransformer: {e}")
                raise
        return self._sentence_transformer
    
    def get_or_create_llm(self, provider: str, model_id: str = None, **kwargs):
        """
        Get or create a cached LLM instance.
        Reduces overhead of creating identical LLM clients repeatedly.
        
        Args:
            provider: LLM provider name
            model_id: Model identifier
            **kwargs: Additional model parameters
            
        Returns:
            Cached or new LLM instance
        """
        cache_key = f"{provider}:{model_id or 'default'}"
        
        if cache_key in self._llm_cache:
            logger.debug(f"[SERVICE] ♻️  Reusing cached LLM: {cache_key}")
            return self._llm_cache[cache_key]
        
        logger.debug(f"[SERVICE] 🔨 Creating new LLM instance: {cache_key}")
        from services.llm.factory import LLMFactory
        
        llm_kwargs = {}
        if model_id:
            llm_kwargs["model"] = model_id
        llm_kwargs.update(kwargs)
        
        try:
            llm_instance = LLMFactory.create(provider=provider, **llm_kwargs)
            if llm_instance.is_available():
                self._llm_cache[cache_key] = llm_instance
        except Exception as e:
            logger.error(f"[SERVICE] ❌ Failed to create LLM instance {cache_key}: {e}")
            raise
        
        return llm_instance
    
    def get_tiered_llm(self, tier: str):
        """
        Get an LLM instance for a specific task tier.
        Uses model_tiers configuration for optimal model selection.
        
        Args:
            tier: One of "intent", "extraction", "reasoning"
            
        Returns:
            LLM instance configured for the tier
        """
        from core.model_tiers import get_model_for_tier
        
        tier_config = get_model_for_tier(tier)
        provider = tier_config["provider"]
        model_id = tier_config.get("model_id")
        
        logger.debug(f"[SERVICE] Getting tiered LLM for {tier}: {provider}/{model_id or 'default'}")
        return self.get_or_create_llm(provider, model_id)
    
    def is_initialized(self) -> bool:
        """Check if services are initialized."""
        return self._initialized
    
    def get_mode_info(self) -> dict:
        """Get information about current service configuration."""
        return {
            "run_mode": Config.APP_RUN_MODE,
            "use_local_llm": Config.USE_LOCAL_LLM,
            "use_local_classifier": Config.USE_LOCAL_CLASSIFIER,
            "llm_provider": os.getenv("LLM_PROVIDER", "auto"),
            "initialized": self._initialized,
            "llm_available": self._llm_client.is_available() if self._llm_client else False,
            "classifier_available": self._classifier.is_available() if self._classifier else False,
        }
    
    def shutdown(self) -> None:
        """Shutdown and cleanup services."""
        logger.info("[SERVICE] Shutting down services...")
        
        # Clear cache on LLM if available
        if self._llm_client and hasattr(self._llm_client, 'clear_cache'):
            self._llm_client.clear_cache()
        
        # Clear LLM cache
        for llm in self._llm_cache.values():
            if hasattr(llm, 'clear_cache'):
                llm.clear_cache()
        
        self._llm_client = None
        self._classifier = None
        self._search_client = None
        self._pipeline_orchestrator = None
        self._ocr_service = None
        self._keybert_model = None
        self._sentence_transformer = None
        self._llm_cache.clear()
        self._initialized = False
        
        logger.info("[SERVICE] Services shutdown complete")


# Global service manager instance
service_manager = ServiceManager()
