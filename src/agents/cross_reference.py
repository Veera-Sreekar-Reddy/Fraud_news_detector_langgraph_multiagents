"""
Cross Reference Agent
Responsible for cross-referencing with multiple sources
"""
from typing import Dict, Any, List
from ..core.base_agent import BaseAgent
from ..core.state import AgentState
from ..core.exceptions import AgentProcessingError
from ..config import get_config


class CrossReferenceAgent(BaseAgent):
    """Agent responsible for cross-referencing with multiple sources"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Cross Reference Agent
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__("CrossReferenceAgent", "cross_referencing")
        self.config = get_config()
    
    def _analyze_consensus(self, fact_check_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze consensus among fact-checkers
        
        Args:
            fact_check_results: List of fact-check results
            
        Returns:
            Consensus analysis dictionary
        """
        if not fact_check_results:
            return {
                "consensus": False,
                "consensus_level": "none",
                "verdicts": []
            }
        
        verdicts = [result.get("verdict", "Unknown") for result in fact_check_results]
        unique_verdicts = set(verdicts)
        
        consensus = len(unique_verdicts) == 1
        consensus_level = "high" if len(fact_check_results) > 2 else "medium" if len(fact_check_results) > 1 else "low"
        
        return {
            "consensus": consensus,
            "consensus_level": consensus_level,
            "verdicts": verdicts,
            "unique_verdicts": list(unique_verdicts),
            "total_sources": len(fact_check_results)
        }
    
    def _create_cross_references(self, fact_check_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create cross-reference entries
        
        Args:
            fact_check_results: List of fact-check results
            
        Returns:
            List of cross-reference dictionaries
        """
        cross_references = []
        
        for result in fact_check_results:
            cross_references.append({
                "source": result.get("source", "Unknown"),
                "verdict": result.get("verdict", "Unknown"),
                "confidence": result.get("confidence", 0.0),
                "matches_other_sources": len(fact_check_results) > 1
            })
        
        return cross_references
    
    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Process state and cross-reference sources
        
        Args:
            state: Current agent state
            
        Returns:
            Dictionary with state updates
            
        Raises:
            AgentProcessingError: If processing fails
        """
        try:
            self.logger.info("Cross-referencing with multiple sources")
            fact_check_results = state.get("fact_check_results", [])
            search_results = state.get("search_results", [])
            
            # Create cross-references
            cross_references = self._create_cross_references(fact_check_results)
            
            # Analyze consensus
            consensus_analysis = self._analyze_consensus(fact_check_results)
            
            consensus = consensus_analysis["consensus"]
            
            self.logger.info(
                f"Cross-referenced {len(cross_references)} sources, "
                f"consensus: {consensus} ({consensus_analysis['consensus_level']})"
            )
            
            # Send message to LogicalAnalyzer
            message = self.send_message(
                to_agent="LogicalAnalyzer",
                message_type="cross_reference",
                content={
                    "cross_references": cross_references,
                    "consensus": consensus,
                    "consensus_analysis": consensus_analysis
                },
                confidence=0.8
            )
            
            decision = self.log_decision(
                decision=f"Found {len(cross_references)} cross-references",
                reasoning=f"Consensus: {consensus} ({consensus_analysis['consensus_level']})",
                confidence=0.8
            )
            
            return {
                "cross_references": cross_references,
                "agent_messages": [message],  # Reducer will merge
                "agent_decisions": {self.agent_name: decision}  # Reducer will merge
            }
        except Exception as e:
            error_msg = f"Failed to cross-reference: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise AgentProcessingError(error_msg, agent_name=self.agent_name) from e

