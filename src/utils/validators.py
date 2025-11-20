"""
Validation utilities for the multi-agent system
"""
from typing import Optional
from urllib.parse import urlparse
from ..core.exceptions import ValidationError


def validate_url(url: str) -> bool:
    """
    Validate URL format
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If URL is invalid
    """
    if not url or not url.strip():
        raise ValidationError("URL cannot be empty", field="url")
    
    url = url.strip()
    
    try:
        result = urlparse(url)
        if not result.scheme:
            raise ValidationError(
                "URL must include scheme (http:// or https://)",
                field="url"
            )
        if not result.netloc:
            raise ValidationError("URL must include domain", field="url")
        return True
    except Exception as e:
        raise ValidationError(f"Invalid URL format: {str(e)}", field="url") from e


def validate_query(query: str) -> bool:
    """
    Validate query string
    
    Args:
        query: Query to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If query is invalid
    """
    if not query or not query.strip():
        raise ValidationError("Query cannot be empty", field="query")
    
    if len(query.strip()) < 3:
        raise ValidationError(
            "Query must be at least 3 characters long",
            field="query"
        )
    
    if len(query.strip()) > 10000:
        raise ValidationError(
            "Query must be less than 10000 characters",
            field="query"
        )
    
    return True


def validate_credibility_score(score: int) -> bool:
    """
    Validate credibility score
    
    Args:
        score: Score to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If score is invalid
    """
    if not isinstance(score, int):
        raise ValidationError(
            "Credibility score must be an integer",
            field="credibility_score"
        )
    
    if not (0 <= score <= 100):
        raise ValidationError(
            "Credibility score must be between 0 and 100",
            field="credibility_score"
        )
    
    return True


def validate_confidence_score(score: float) -> bool:
    """
    Validate confidence score
    
    Args:
        score: Score to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If score is invalid
    """
    if not isinstance(score, (int, float)):
        raise ValidationError(
            "Confidence score must be a number",
            field="confidence_score"
        )
    
    if not (0.0 <= score <= 1.0):
        raise ValidationError(
            "Confidence score must be between 0.0 and 1.0",
            field="confidence_score"
        )
    
    return True

