"""
Llama3.3 70B API Client
Handles API calls to Llama3.3 70B model
"""
import os
import logging
from typing import Dict, Any, Optional, List
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class Llama3Client:
    """Client for interacting with Llama3.3 70B API"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: str = "llama-3.3-70b",
        timeout: int = 60
    ):
        """
        Initialize Llama3 API client
        
        Args:
            api_key: API key for Llama3 service (defaults to LLAMA3_API_KEY env var)
            api_url: API endpoint URL (defaults to LLAMA3_API_URL env var)
            model: Model name (defaults to llama-3.3-70b)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("LLAMA3_API_KEY")
        # Groq API endpoint (OpenAI-compatible)
        default_url = os.getenv("LLAMA3_API_URL", "https://api.groq.com/openai/v1/chat/completions")
        self.api_url = api_url or default_url
        # Groq model name (can be overridden via env var)
        default_model = os.getenv("LLAMA3_MODEL", "llama-3.3-70b-versatile")
        self.model = model if model != "llama-3.3-70b" else default_model
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError(
                "LLAMA3_API_KEY not found. Please set it in .env file or pass as parameter."
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text using Llama3.3 70B
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If API call fails
        """
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p
            }
            
            logger.debug(f"Sending request to Llama3 API: {self.api_url}")
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                logger.debug(f"Received response from Llama3 API: {len(content)} characters")
                return content
            else:
                raise ValueError("Invalid response format from Llama3 API")
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Llama3 API request failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to generate text with Llama3: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    def analyze_claim(
        self,
        claim: str,
        context: Optional[str] = None,
        analysis_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Analyze a claim using Llama3.3 70B
        
        Args:
            claim: The claim to analyze
            context: Optional context information
            analysis_type: Type of analysis (fact_check, sentiment, logical, etc.)
            
        Returns:
            Analysis results as dictionary
        """
        system_prompts = {
            "fact_check": """You are an expert fact-checker and disinformation analyst. Your task is to analyze claims based on:
1. Source credibility and patterns
2. Logical consistency and plausibility
3. Known disinformation patterns and red flags
4. Claim structure and language patterns

DO NOT say you cannot verify due to lack of real-time information. Instead, analyze based on:
- Whether the claim matches known patterns of misinformation
- The credibility and reputation of the source domain
- Logical consistency of the claim
- Presence of manipulative language or exaggeration
- Alignment with established scientific/historical facts you know

Provide a verdict: TRUE, FALSE, MISLEADING, or UNVERIFIABLE
- TRUE: Claim appears credible and aligns with known facts/patterns
- FALSE: Claim contradicts known facts or shows clear disinformation patterns
- MISLEADING: Claim contains partial truth but is distorted or exaggerated
- UNVERIFIABLE: Cannot determine with available patterns/analysis (use sparingly)

IMPORTANT: You MUST provide a verdict. Do NOT default to UNVERIFIABLE unless absolutely impossible to determine.
- If the source is low credibility (<20) and claim is sensational: likely FALSE
- If source is high credibility (>80) and claim is reasonable: likely TRUE
- If claim has logical inconsistencies or contradictions: likely FALSE or MISLEADING
- If claim uses manipulative language or exaggeration: likely MISLEADING

Respond ONLY in valid JSON format with these exact keys:
{
  "verdict": "TRUE" or "FALSE" or "MISLEADING" or "UNVERIFIABLE",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation based on source patterns, logic, and known information",
  "evidence": "Supporting observations or patterns detected"
}""",
            
            "sentiment": """You are a sentiment and emotional manipulation detection expert. Analyze the text for:
1. Sentiment: positive, negative, or neutral
2. Emotional manipulation indicators
3. Manipulative language patterns
4. Exaggeration or sensationalism

Look for patterns like: urgency, fear-mongering, clickbait language, superlatives, emotional appeals.

Respond ONLY in valid JSON format:
{
  "sentiment": "positive" or "negative" or "neutral",
  "is_emotional_appeal": true or false,
  "manipulative_phrases": ["phrase1", "phrase2"],
  "manipulation_score": 0.0-1.0
}""",
            
            "logical": """You are a logical fallacy and contradiction detection expert. Analyze the claim for:
1. Logical fallacies (false dilemma, ad hominem, slippery slope, appeal to emotion, etc.)
2. Internal contradictions
3. Reasoning quality

Detect fallacies based on claim structure and language patterns.

Respond ONLY in valid JSON format:
{
  "fallacies": ["fallacy_type1", "fallacy_type2"],
  "has_contradictions": true or false,
  "reasoning_quality": "good" or "fair" or "poor"
}""",
            
            "general": """You are an expert disinformation analyst. Analyze claims based on patterns, source credibility, logical consistency, and known disinformation tactics. 
Provide your analysis based on what you can determine from the claim structure, source, and logical patterns.

Respond in valid JSON format with relevant analysis fields."""
        }
        
        system_prompt = system_prompts.get(analysis_type, system_prompts["general"])
        
        user_prompt = f"Analyze this claim: \"{claim}\"\n"
        if context:
            user_prompt += f"\nAdditional context:\n{context}\n"
        
        if analysis_type == "fact_check":
            user_prompt += """
CRITICAL: You MUST analyze and provide a verdict based on available information. DO NOT default to UNVERIFIABLE.

Analysis approach:
1. Evaluate source credibility - low credibility domains often publish false claims
2. Check claim structure - sensational or extreme claims from low-credibility sources are often FALSE
3. Look for logical inconsistencies - contradictory statements indicate FALSE
4. Assess plausibility - does the claim align with known patterns and facts?
5. Check for manipulative language - exaggeration or emotional manipulation suggests MISLEADING

Examples:
- Sensational claim ("cure discovered today") + low credibility source = likely FALSE
- Reasonable claim + high credibility source (reuters.com, bbc.com) = likely TRUE  
- Claim with logical contradictions = likely FALSE
- Partially true but exaggerated = likely MISLEADING

Provide your verdict based on these patterns, not on needing real-time verification."""
        else:
            user_prompt += "\nProvide your analysis in valid JSON format."
        
        try:
            response = self.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.5,  # Slightly higher for more nuanced analysis
                max_tokens=800  # Increased for more detailed reasoning
            )
            
            # Try to parse JSON from response
            import json
            import re
            
            # Extract JSON from response (handle cases where LLM adds text before/after JSON)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            result = json.loads(response)
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}. Using raw response.")
            return {
                "raw_response": response,
                "error": "Failed to parse JSON response"
            }
        except Exception as e:
            logger.error(f"Failed to analyze claim: {e}", exc_info=True)
            raise


# Global client instance
_llama3_client: Optional[Llama3Client] = None


def get_llama3_client() -> Llama3Client:
    """Get or create global Llama3 client instance"""
    global _llama3_client
    
    if _llama3_client is None:
        _llama3_client = Llama3Client()
    
    return _llama3_client

