"""
Abstract interface for classification models.
Defines the contract that all classifier implementations must follow.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Union


class ClassifierInterface(ABC):
    """Abstract base class for classification models."""
    
    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classify the given text and return prediction results.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary containing prediction results with at least:
            - 'label': predicted class label
            - 'confidence': confidence score (0.0 to 1.0)
            
        Raises:
            ClassifierError: If classification fails
        """
        pass
    
    @abstractmethod
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple texts in a batch.
        
        Args:
            texts: List of input texts to classify
            
        Returns:
            List of prediction dictionaries
            
        Raises:
            ClassifierError: If batch classification fails
        """
        pass
    
    @abstractmethod
    def get_supported_labels(self) -> List[str]:
        """
        Get list of supported classification labels.
        
        Returns:
            List of supported labels
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the classifier is available and ready for use.
        
        Returns:
            True if available, False otherwise
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the classification model.
        
        Returns:
            Dictionary containing model metadata
        """
        pass