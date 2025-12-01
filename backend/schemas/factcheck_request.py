"""
Request schemas for fact-checking endpoints.
Defines the structure and validation for fact-check API requests.
"""
from typing import Optional, List, Dict, Any


class FactCheckRequest:
    """Schema for fact-check request payload."""
    
    def __init__(self, data: Dict[str, Any]):
        self.text = data.get("text", "").strip()
        self.source_url = data.get("source_url")
        self.enable_search = data.get("enable_search", True)
        self.max_search_results = data.get("max_search_results", 10)
    
    def validate(self) -> Optional[str]:
        """Validate the request data."""
        if not self.text:
            return "Text field is required and cannot be empty"
        
        if len(self.text) > 5000:
            return "Text field cannot exceed 5000 characters"
        
        if self.max_search_results < 1 or self.max_search_results > 20:
            return "max_search_results must be between 1 and 20"
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'text': self.text,
            'source_url': self.source_url,
            'enable_search': self.enable_search,
            'max_search_results': self.max_search_results
        }


class BulkFactCheckRequest:
    """Schema for bulk fact-check request payload."""
    
    def __init__(self, data: Dict[str, Any]):
        self.claims = data.get("claims", [])
        self.enable_search = data.get("enable_search", True)
        self.max_search_results = data.get("max_search_results", 5)
    
    def validate(self) -> Optional[str]:
        """Validate the request data."""
        if not self.claims:
            return "Claims list is required and cannot be empty"
        
        if len(self.claims) > 10:
            return "Cannot process more than 10 claims at once"
        
        for i, claim in enumerate(self.claims):
            if not claim or not claim.strip():
                return f"Claim {i+1} cannot be empty"
        
        if self.max_search_results < 1 or self.max_search_results > 10:
            return "max_search_results must be between 1 and 10"
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'claims': self.claims,
            'enable_search': self.enable_search,
            'max_search_results': self.max_search_results
        }
