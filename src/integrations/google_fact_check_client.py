"""
Google Fact Check API Client
Handles API calls to Google Fact Check Tools API
"""
import os
import logging
from typing import Dict, Any, Optional, List
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class GoogleFactCheckClient:
    """Client for interacting with Google Fact Check API"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize Google Fact Check API client
        
        Args:
            api_key: API key for Google Fact Check (defaults to GOOGLE_FACT_CHECK_API_KEY env var)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("GOOGLE_FACT_CHECK_API_KEY")
        self.timeout = timeout
        self.base_url = "https://factchecktools.googleapis.com/v1alpha1"
        
        if not self.api_key:
            raise ValueError(
                "GOOGLE_FACT_CHECK_API_KEY not found. Please set it in .env file or pass as parameter."
            )
    
    def search_claims(
        self,
        query: str,
        language_code: str = "en",
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for fact-checked claims
        
        Args:
            query: Claim or statement to search for
            language_code: Language code (default: "en")
            max_results: Maximum number of results to return
            
        Returns:
            List of fact-check results
            
        Raises:
            Exception: If API call fails
        """
        try:
            url = f"{self.base_url}/claims:search"
            params = {
                "key": self.api_key,
                "query": query,
                "languageCode": language_code,
                "maxAgeDays": 365,  # Search within last year
                "pageSize": max_results
            }
            
            logger.debug(f"Searching Google Fact Check API for: {query[:50]}...")
            
            response = requests.get(
                url,
                params=params,
                timeout=self.timeout
            )
            
            # Check for errors and log response details
            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"API Error {response.status_code}: {error_detail}")
                # Try to parse error JSON
                try:
                    error_json = response.json()
                    error_msg = error_json.get("error", {}).get("message", error_detail)
                    raise Exception(f"Google Fact Check API error: {error_msg}")
                except:
                    raise Exception(f"Google Fact Check API error: {error_detail}")
            
            response.raise_for_status()
            result = response.json()
            
            # Extract claims from response
            claims = result.get("claims", [])
            logger.info(f"Found {len(claims)} fact-check results for query")
            
            # Format results
            formatted_results = []
            for claim in claims:
                # Get the best rating (most recent or highest confidence)
                claim_data = claim.get("claim", {})
                text = claim_data.get("text", "")
                
                # Get ratings
                ratings = []
                for review in claim.get("claimReview", []):
                    publisher = review.get("publisher", {})
                    rating = review.get("textualRating", "")
                    url = review.get("url", "")
                    review_date = review.get("reviewDate", "")
                    
                    ratings.append({
                        "source": publisher.get("name", "Unknown"),
                        "site": publisher.get("site", ""),
                        "verdict": self._normalize_verdict(rating),
                        "rating": rating,
                        "url": url,
                        "review_date": review_date
                    })
                
                if ratings:
                    # Use the most recent rating or first one
                    best_rating = ratings[0]
                    formatted_results.append({
                        "claim_text": text,
                        "source": best_rating["source"],
                        "verdict": best_rating["verdict"],
                        "confidence": self._calculate_confidence(best_rating["verdict"]),
                        "url": best_rating["url"],
                        "review_date": best_rating["review_date"],
                        "all_ratings": ratings
                    })
            
            return formatted_results
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Google Fact Check API request failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to search fact-check claims: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    def _normalize_verdict(self, rating: str) -> str:
        """
        Normalize verdict from various formats to standard format
        
        Args:
            rating: Raw rating text from API
            
        Returns:
            Normalized verdict: "TRUE", "FALSE", "MISLEADING", or "UNVERIFIABLE"
        """
        rating_lower = rating.lower()
        
        # Map common rating patterns
        if any(word in rating_lower for word in ["false", "incorrect", "wrong", "untrue", "debunked"]):
            return "FALSE"
        elif any(word in rating_lower for word in ["true", "correct", "accurate", "verified"]):
            return "TRUE"
        elif any(word in rating_lower for word in ["misleading", "partially", "mostly", "exaggerated", "distorted"]):
            return "MISLEADING"
        elif any(word in rating_lower for word in ["unproven", "unverified", "unsubstantiated", "cannot verify"]):
            return "UNVERIFIABLE"
        else:
            # Default to MISLEADING if unclear
            return "MISLEADING"
    
    def _calculate_confidence(self, verdict: str) -> float:
        """
        Calculate confidence score based on verdict
        
        Args:
            verdict: Normalized verdict
            
        Returns:
            Confidence score (0.0-1.0)
        """
        # Higher confidence for clear TRUE/FALSE, lower for MISLEADING/UNVERIFIABLE
        confidence_map = {
            "TRUE": 0.85,
            "FALSE": 0.85,
            "MISLEADING": 0.70,
            "UNVERIFIABLE": 0.50
        }
        return confidence_map.get(verdict, 0.60)
    
    def check_claim(
        self,
        claim: str,
        source_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check a claim against Google Fact Check database
        
        Args:
            claim: Claim to check
            source_url: Optional source URL for context
            
        Returns:
            Dictionary with fact-check results
        """
        try:
            # Search for the claim
            results = self.search_claims(claim, max_results=5)
            
            if not results:
                return {
                    "found": False,
                    "verdict": "UNVERIFIABLE",
                    "confidence": 0.40,
                    "reasoning": "No fact-check results found in database",
                    "results": []
                }
            
            # Aggregate results
            verdicts = [r["verdict"] for r in results]
            false_count = verdicts.count("FALSE")
            true_count = verdicts.count("TRUE")
            misleading_count = verdicts.count("MISLEADING")
            
            # Determine overall verdict
            if false_count > 0:
                overall_verdict = "FALSE"
                confidence = 0.80 + (false_count * 0.05)  # Higher confidence with more sources
            elif misleading_count > 0:
                overall_verdict = "MISLEADING"
                confidence = 0.65 + (misleading_count * 0.05)
            elif true_count > 0:
                overall_verdict = "TRUE"
                confidence = 0.75 + (true_count * 0.05)
            else:
                overall_verdict = "UNVERIFIABLE"
                confidence = 0.50
            
            # Cap confidence at 0.95
            confidence = min(confidence, 0.95)
            
            reasoning = f"Found {len(results)} fact-check result(s): {false_count} FALSE, {true_count} TRUE, {misleading_count} MISLEADING"
            
            return {
                "found": True,
                "verdict": overall_verdict,
                "confidence": confidence,
                "reasoning": reasoning,
                "results": results,
                "source_count": len(results)
            }
            
        except Exception as e:
            logger.error(f"Failed to check claim: {e}", exc_info=True)
            return {
                "found": False,
                "verdict": "UNVERIFIABLE",
                "confidence": 0.30,
                "reasoning": f"Error checking claim: {str(e)}",
                "results": []
            }


# Global client instance
_google_fact_check_client: Optional[GoogleFactCheckClient] = None


def get_google_fact_check_client() -> GoogleFactCheckClient:
    """Get or create global Google Fact Check client instance"""
    global _google_fact_check_client
    
    if _google_fact_check_client is None:
        _google_fact_check_client = GoogleFactCheckClient()
    
    return _google_fact_check_client

