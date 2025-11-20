"""
Supervisor Agent
Responsible for coordinating agent activities
"""
from typing import Dict, Any
from ..core.base_agent import BaseAgent
from ..core.state import AgentState
from ..core.exceptions import AgentProcessingError
from ..config import get_config


class SupervisorAgent(BaseAgent):
    """Supervisor agent that coordinates all other agents"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Supervisor Agent
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__("Supervisor", "coordination")
        self.config = get_config()
    
    def _analyze_coordination(self, state: AgentState) -> Dict[str, Any]:
        """
        Analyze coordination status
        
        Args:
            state: Current agent state
            
        Returns:
            Coordination analysis dictionary
        """
        agent_decisions = state.get("agent_decisions", {})
        agent_messages = state.get("agent_messages", [])
        messages_for_supervisor = [
            msg for msg in agent_messages
            if isinstance(msg, dict) and msg.get("to_agent") == "Supervisor"
        ]
        
        coordination_summary = {
            "total_agents_active": len(agent_decisions),
            "messages_received": len(messages_for_supervisor),
            "coordination_status": "successful",
            "agents": list(agent_decisions.keys())
        }
        
        # Determine if additional review is needed
        requires_review = state.get("requires_review", False)
        
        if requires_review:
            coordination_summary["action_required"] = "review_needed"
            coordination_summary["reason"] = state.get("review_reason", "Unknown")
        else:
            coordination_summary["action_required"] = "none"
        
        return coordination_summary
    
    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Process state and coordinate agents
        
        Args:
            state: Current agent state
            
        Returns:
            Dictionary with state updates
            
        Raises:
            AgentProcessingError: If processing fails
        """
        try:
            self.logger.info("Coordinating agent activities")
            
            # Analyze coordination
            coordination_summary = self._analyze_coordination(state)
            
            self.logger.info(
                f"Coordination complete: {coordination_summary['total_agents_active']} agents, "
                f"{coordination_summary['messages_received']} messages"
            )
            
            if coordination_summary["action_required"] == "review_needed":
                self.logger.warning(f"Review required: {coordination_summary['reason']}")
            
            return {
                "agent_decisions": state.get("agent_decisions", {}),
                "workflow_stage": state.get("workflow_stage", "unknown")
            }
        except Exception as e:
            error_msg = f"Failed to coordinate agents: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise AgentProcessingError(error_msg, agent_name=self.agent_name) from e

