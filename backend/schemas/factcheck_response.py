"""
Response schemas for fact-checking endpoints.
Defines the structure for fact-check API responses.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime


class EvidenceItem:
    """Schema for individual evidence item."""
    
    def __init__(
        self,
        title: str,
        url: str,
        snippet: str,
        relevance_score: float,
        source_domain: str
    ):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.relevance_score = relevance_score
        self.source_domain = source_domain
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'title': self.title,
            'url': self.url,
            'snippet': self.snippet,
            'relevance_score': self.relevance_score,
            'source_domain': self.source_domain
        }


class FactCheckResponse:
    """Schema for fact-check response."""
    
    def __init__(
        self,
        claim: str,
        prediction: str,
        confidence: float,
        evidence: List[EvidenceItem],
        processing_time: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ):
        self.claim = claim
        self.prediction = prediction
        self.confidence = confidence
        self.evidence = evidence
        self.processing_time = processing_time
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'claim': self.claim,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'evidence': [item.to_dict() for item in self.evidence],
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class BulkFactCheckResponse:
    """Schema for bulk fact-check response."""
    
    def __init__(
        self,
        results: List[FactCheckResponse],
        total_processing_time: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ):
        self.results = results
        self.total_processing_time = total_processing_time
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'results': [result.to_dict() for result in self.results],
            'total_processing_time': self.total_processing_time,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class ErrorResponse:
    """Schema for error responses."""
    
    def __init__(
        self,
        error: str,
        message: str,
        status_code: int,
        timestamp: Optional[datetime] = None
    ):
        self.error = error
        self.message = message
        self.status_code = status_code
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'error': self.error,
            'message': self.message,
            'status_code': self.status_code,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
