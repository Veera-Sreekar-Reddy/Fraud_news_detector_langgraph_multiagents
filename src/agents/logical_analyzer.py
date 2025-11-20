"""
Logical Analyzer Agent
Responsible for logical analysis and fallacy detection
"""
from typing import Dict, Any, List
import re
from ..core.base_agent import BaseAgent
from ..core.state import AgentState
from ..core.exceptions import AgentProcessingError
from ..config import get_config
from ..integrations import get_llama3_client


class LogicalAnalyzerAgent(BaseAgent):
    """Agent responsible for logical analysis and fallacy detection"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Logical Analyzer Agent
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__("LogicalAnalyzer", "logical_analysis")
        self.config = get_config()
        self.fallacy_patterns = (
            config.get("fallacy_patterns", {}) if config
            else self.config.fallacy_patterns
        )
        
        try:
            self.llama3_client = get_llama3_client()
            self.use_llama3 = True
        except Exception as e:
            self.logger.warning(f"Llama3 API not available, using fallback: {e}")
            self.llama3_client = None
            self.use_llama3 = False
    
    def _detect_fallacies(self, query: str) -> List[str]:
        """
        Detect logical fallacies in query using Llama3.3 70B
        
        Args:
            query: Claim query
            
        Returns:
            List of detected fallacies
        """
        # Use Llama3 API if available
        if self.use_llama3 and self.llama3_client:
            try:
                self.logger.info("Using Llama3.3 70B for logical analysis")
                
                analysis = self.llama3_client.analyze_claim(
                    claim=query,
                    analysis_type="logical"
                )
                
                if isinstance(analysis, dict) and "fallacies" in analysis:
                    fallacies = analysis.get("fallacies", [])
                    if isinstance(fallacies, list):
                        return fallacies
                    elif isinstance(fallacies, str):
                        return [fallacies] if fallacies else []
                        
            except Exception as e:
                self.logger.warning(f"Llama3 API call failed, using fallback: {e}")
                # Fall through to fallback
        
        # Fallback to rule-based detection
        self.logger.info("Using fallback fallacy detection")
        query_lower = query.lower()
        fallacies = []
        
        for fallacy_type, pattern in self.fallacy_patterns.items():
            if re.search(pattern, query_lower, re.IGNORECASE):
                fallacies.append(fallacy_type)
        
        return fallacies
    
    def _check_contradictions(
        self,
        credibility_score: int,
        fact_check_results: List[Dict[str, Any]]
    ) -> tuple[bool, str]:
        """
        Check for contradictions
        
        Args:
            credibility_score: Source credibility score
            fact_check_results: List of fact-check results
            
        Returns:
            Tuple of (has_contradiction, reasoning)
        """
        contradiction = False
        reasoning = ""
        
        # Check low credibility
        if credibility_score < self.config.low_credibility_threshold:
            contradiction = True
            reasoning = "Low credibility source contradicts high-credibility fact-checkers"
        
        # Check fact-check results
        if fact_check_results:
            false_count = sum(
                1 for r in fact_check_results
                if r.get("verdict") == "False"
            )
            if false_count > 0:
                contradiction = True
                if reasoning:
                    reasoning += ". "
                reasoning += "Fact-checkers contradict the claim"
        
        if not contradiction:
            reasoning = "No major contradictions found"
        
        return contradiction, reasoning
    
    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Process state and analyze logic
        
        Args:
            state: Current agent state
            
        Returns:
            Dictionary with state updates
            
        Raises:
            AgentProcessingError: If processing fails
        """
        try:
            self.logger.info("Analyzing logical consistency and fallacies")
            query = state.get("query", "")
            credibility_score = state.get("credibility_score", 50)
            fact_check_results = state.get("fact_check_results", [])
            
            if not query:
                raise AgentProcessingError("Query is empty", agent_name=self.agent_name)
            
            # Detect logical fallacies
            fallacies = self._detect_fallacies(query)
            
            # Check for contradictions
            contradiction, reasoning = self._check_contradictions(
                credibility_score,
                fact_check_results
            )
            
            # Receive messages from other agents
            evidence_messages = self.receive_messages(state, "evidence")
            sentiment_messages = self.receive_messages(state, "sentiment")
            
            self.logger.info(
                f"Found {len(fallacies)} fallacies, "
                f"contradiction: {contradiction}"
            )
            
            internal_contradiction = contradiction
            
            # Send message to Supervisor
            message = self.send_message(
                to_agent="Supervisor",
                message_type="logical_analysis",
                content={
                    "fallacies": fallacies,
                    "contradiction": contradiction,
                    "reasoning": reasoning
                },
                confidence=0.85
            )
            
            decision = self.log_decision(
                decision=f"Found {len(fallacies)} fallacies, Contradiction: {contradiction}",
                reasoning=reasoning,
                confidence=0.85
            )
            
            return {
                "logical_fallacies": fallacies,
                "internal_contradiction": internal_contradiction,
                "workflow_stage": "analyzed",
                "agent_messages": [message],  # Reducer will merge
                "agent_decisions": {self.agent_name: decision}  # Reducer will merge
            }
        except Exception as e:
            error_msg = f"Failed to analyze logic: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise AgentProcessingError(error_msg, agent_name=self.agent_name) from e

