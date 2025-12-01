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
        self._llm_client = LLMFactory.create()
        
        if self._llm_client.is_available():
            info = self._llm_client.get_model_info()
            logger.info(f"[SERVICE] LLM initialized: {info.get('provider', 'unknown')}")
        else:
            logger.warning("[SERVICE] LLM client initialized but not available")
    
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
        
        self._llm_client = None
        self._classifier = None
        self._search_client = None
        self._initialized = False
        
        logger.info("[SERVICE] Services shutdown complete")


# Global service manager instance
service_manager = ServiceManager()
