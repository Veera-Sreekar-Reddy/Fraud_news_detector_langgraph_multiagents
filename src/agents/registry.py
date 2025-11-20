"""
Agent Registry
Manages agent instances and provides factory methods
"""
from typing import Dict, Any, Optional, Type
from ..core.base_agent import BaseAgent
from ..core.exceptions import AgentError
from .triage_manager import TriageManagerAgent
from .source_scorer import SourceScorerAgent
from .evidence_gatherer import EvidenceGathererAgent
from .sentiment_analyzer import SentimentAnalyzerAgent
from .cross_reference import CrossReferenceAgent
from .logical_analyzer import LogicalAnalyzerAgent
from .verdict_synthesizer import VerdictSynthesizerAgent
from .supervisor import SupervisorAgent


class AgentRegistry:
    """Registry for managing agent instances"""
    
    def __init__(self):
        """Initialize agent registry"""
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_classes: Dict[str, Type[BaseAgent]] = {
            "TriageManager": TriageManagerAgent,
            "SourceScorer": SourceScorerAgent,
            "EvidenceGatherer": EvidenceGathererAgent,
            "SentimentAnalyzer": SentimentAnalyzerAgent,
            "CrossReferenceAgent": CrossReferenceAgent,
            "LogicalAnalyzer": LogicalAnalyzerAgent,
            "VerdictSynthesizer": VerdictSynthesizerAgent,
            "Supervisor": SupervisorAgent,
        }
    
    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """
        Register an agent instance
        
        Args:
            name: Agent name
            agent: Agent instance
            
        Raises:
            AgentError: If registration fails
        """
        if not name or not name.strip():
            raise AgentError("Agent name cannot be empty")
        
        if not isinstance(agent, BaseAgent):
            raise AgentError("Agent must be an instance of BaseAgent")
        
        self._agents[name.strip()] = agent
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """
        Get agent instance by name
        
        Args:
            name: Agent name
            
        Returns:
            Agent instance or None if not found
        """
        return self._agents.get(name.strip())
    
    def create_agent(self, name: str, config: Optional[Dict[str, Any]] = None) -> BaseAgent:
        """
        Create agent instance by name
        
        Args:
            name: Agent name
            config: Optional configuration dictionary
            
        Returns:
            Agent instance
            
        Raises:
            AgentError: If agent creation fails
        """
        name = name.strip()
        
        if name not in self._agent_classes:
            raise AgentError(f"Unknown agent type: {name}")
        
        agent_class = self._agent_classes[name]
        
        try:
            if config:
                agent = agent_class(config=config)
            else:
                agent = agent_class()
            
            self.register_agent(name, agent)
            return agent
        except Exception as e:
            raise AgentError(f"Failed to create agent {name}: {str(e)}") from e
    
    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """
        Get all registered agents
        
        Returns:
            Dictionary of agent instances
        """
        return self._agents.copy()
    
    def clear(self) -> None:
        """Clear all registered agents"""
        self._agents.clear()
    
    def register_agent_class(self, name: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register a new agent class
        
        Args:
            name: Agent name
            agent_class: Agent class
            
        Raises:
            AgentError: If registration fails
        """
        if not name or not name.strip():
            raise AgentError("Agent name cannot be empty")
        
        if not issubclass(agent_class, BaseAgent):
            raise AgentError("Agent class must be a subclass of BaseAgent")
        
        self._agent_classes[name.strip()] = agent_class


# Global registry instance
_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    """Get global agent registry instance"""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry

