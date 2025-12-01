"""
Abstract interface for search clients.
Defines the contract that all search implementations must follow.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class SearchInterface(ABC):
    """Abstract base class for search clients."""
    
    @abstractmethod
    def search(
        self, 
        query: str, 
        num_results: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform a search query and return results.
        
        Args:
            query: Search query string
            num_results: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of search result dictionaries containing at least:
            - 'title': result title
            - 'url': result URL
            - 'snippet': result snippet/description
            
        Raises:
            SearchError: If search operation fails
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the search client is available and ready for use.
        
        Returns:
            True if available, False otherwise
        """
        pass
    
    @abstractmethod
    def get_search_info(self) -> Dict[str, Any]:
        """
        Get information about the search service.
        
        Returns:
            Dictionary containing search service metadata
        """
        pass