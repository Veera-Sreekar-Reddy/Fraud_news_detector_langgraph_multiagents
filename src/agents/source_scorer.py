"""
Source Scorer Agent
Responsible for assessing source credibility
"""
from typing import Dict, Any
from urllib.parse import urlparse
from ..core.base_agent import BaseAgent
from ..core.state import AgentState
from ..core.exceptions import AgentProcessingError
from ..config import get_config
from ..utils.validators import validate_url


class SourceScorerAgent(BaseAgent):
    """Agent responsible for assessing source credibility"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Source Scorer Agent
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__("SourceScorer", "credibility_assessment")
        self.config = get_config()
        self.low_credibility_domains = (
            config.get("low_credibility_domains", []) if config
            else self.config.low_credibility_domains
        )
        self.high_credibility_domains = (
            config.get("high_credibility_domains", []) if config
            else self.config.high_credibility_domains
        )
        self.low_threshold = self.config.low_credibility_threshold
        self.high_threshold = self.config.high_credibility_threshold
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc or parsed.path.split("/")[0]
        except Exception:
            return url.replace("http://", "").replace("https://", "").split("/")[0]
    
    def _calculate_credibility_score(self, domain: str) -> tuple[int, str]:
        """
        Calculate credibility score for a domain
        
        Args:
            domain: Domain name
            
        Returns:
            Tuple of (score, reasoning)
        """
        domain_lower = domain.lower()
        
        # Check against known domains
        if any(low_domain.lower() in domain_lower for low_domain in self.low_credibility_domains):
            return 15, "Known low-credibility domain"
        
        if any(high_domain.lower() in domain_lower for high_domain in self.high_credibility_domains):
            return 85, "Known high-credibility domain"
        
        # Analyze URL characteristics
        if ".gov" in domain_lower:
            return 75, "Government domain"
        elif ".edu" in domain_lower:
            return 75, "Educational domain"
        elif "blog" in domain_lower or "wordpress" in domain_lower:
            return 30, "Blog or personal website"
        else:
            return 50, "Unknown domain, neutral credibility"
    
    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Process state and evaluate source credibility
        
        Args:
            state: Current agent state
            
        Returns:
            Dictionary with state updates
            
        Raises:
            AgentProcessingError: If processing fails
        """
        try:
            self.logger.info("Evaluating source credibility")
            source_url = state.get("source_url", "")
            
            if not source_url:
                raise AgentProcessingError("Source URL is empty", agent_name=self.agent_name)
            
            # Validate URL
            try:
                validate_url(source_url)
            except Exception as e:
                self.logger.warning(f"Invalid URL format: {str(e)}")
            
            # Extract domain
            domain = self._extract_domain(source_url)
            self.logger.debug(f"Extracted domain: {domain}")
            
            # Calculate credibility score
            score, reasoning = self._calculate_credibility_score(domain)
            
            # Determine confidence based on score
            confidence = 0.8 if score < self.low_threshold or score > self.high_threshold else 0.6
            
            # Send message to supervisor
            message = self.send_message(
                to_agent="Supervisor",
                message_type="credibility_score",
                content={"score": score, "reasoning": reasoning, "domain": domain},
                confidence=confidence
            )
            
            decision = self.log_decision(
                decision=f"Credibility score: {score}",
                reasoning=reasoning,
                confidence=confidence
            )
            
            self.logger.info(f"Credibility assessment complete: {score}/100 ({reasoning})")
            
            return {
                "credibility_score": score,
                "agent_messages": [message],  # Reducer will merge
                "agent_decisions": {self.agent_name: decision}  # Reducer will merge
            }
        except Exception as e:
            error_msg = f"Failed to process source scoring: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise AgentProcessingError(error_msg, agent_name=self.agent_name) from e

