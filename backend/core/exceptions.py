"""
Custom exception classes used across the FactuAI backend modules.

Provides consistent error handling and messaging throughout the application.
All exceptions inherit from FactuAIException for easy catching of
application-specific errors.
"""


class FactuAIException(Exception):
    """Base exception class for all FactuAI-specific errors."""
    pass


class LLMClientError(FactuAIException):
    """Raised when LLM client operations fail."""
    pass


class ClassifierError(FactuAIException):
    """Raised when classification operations fail."""
    pass


class SearchError(FactuAIException):
    """Raised when search operations fail."""
    pass


class ScrapingError(FactuAIException):
    """Raised when web scraping operations fail."""
    pass


class EvidenceError(FactuAIException):
    """Raised when evidence filtering/ranking operations fail."""
    pass


class ExtractionError(FactuAIException):
    """Raised when claim extraction operations fail."""
    pass


class PipelineError(FactuAIException):
    """Raised when pipeline orchestration fails."""
    pass


class ValidationError(FactuAIException):
    """Raised when input validation fails."""
    pass


class DatabaseError(FactuAIException):
    """Raised when database operations fail."""
    pass


class AuthenticationError(FactuAIException):
    """Raised when authentication operations fail."""
    pass
