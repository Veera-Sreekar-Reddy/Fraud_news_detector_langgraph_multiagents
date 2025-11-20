"""
Triage Manager Agent
Responsible for classifying and categorizing claims
"""
from typing import Dict, Any
from ..core.base_agent import BaseAgent
from ..core.state import AgentState
from ..core.exceptions import AgentProcessingError
from ..config import get_config


class TriageManagerAgent(BaseAgent):
    """Agent responsible for classifying and categorizing claims"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Triage Manager Agent
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__("TriageManager", "classification")
        self.config = get_config()
        self.category_keywords = config.get("category_keywords", {}) if config else self.config.category_keywords
    
    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Process state and classify claim category
        
        Args:
            state: Current agent state
            
        Returns:
            Dictionary with state updates
            
        Raises:
            AgentProcessingError: If processing fails
        """
        try:
            self.logger.info("Analyzing claim category")
            query = state.get("query", "").lower()
            
            if not query:
                raise AgentProcessingError("Query is empty", agent_name=self.agent_name)
            
            # Classify category based on keywords
            category = "general"
            max_matches = 0
            
            for cat, keywords in self.category_keywords.items():
                if cat == "general":
                    continue
                matches = sum(1 for keyword in keywords if keyword in query)
                if matches > max_matches:
                    max_matches = matches
                    category = cat
            
            # Store in memory
            self.store_in_memory("category", category)
            
            # Create message for other agents
            message = self.send_message(
                to_agent="Supervisor",
                message_type="classification",
                content={"category": category, "confidence": 0.9},
                confidence=0.9
            )
            
            decision = self.log_decision(
                decision=f"Classified as {category}",
                reasoning=f"Found {max_matches} matching keywords",
                confidence=0.9
            )
            
            self.logger.info(f"Classification complete: {category} (matches: {max_matches})")
            
            return {
                "category": category,
                "workflow_stage": "triaged",
                "agent_messages": [message],  # Reducer will merge
                "agent_decisions": {self.agent_name: decision}  # Reducer will merge
            }
        except Exception as e:
            error_msg = f"Failed to process triage: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise AgentProcessingError(error_msg, agent_name=self.agent_name) from e

