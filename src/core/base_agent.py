"""
Base Agent Class for Multi-Agent Architecture
Provides common functionality for all specialized agents
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time
import logging
from .state import AgentState, AgentMessage
from .exceptions import AgentError, AgentProcessingError

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system"""
    
    def __init__(self, agent_name: str, agent_type: str):
        """
        Initialize base agent
        
        Args:
            agent_name: Unique name for the agent
            agent_type: Type/category of the agent
        """
        if not agent_name or not agent_name.strip():
            raise ValueError("Agent name cannot be empty")
        if not agent_type or not agent_type.strip():
            raise ValueError("Agent type cannot be empty")
        
        self.agent_name = agent_name.strip()
        self.agent_type = agent_type.strip()
        self.memory: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(f"{__name__}.{self.agent_name}")
        
    def send_message(
        self,
        to_agent: str,
        message_type: str,
        content: Any,
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Create and send a message to another agent
        
        Args:
            to_agent: Recipient agent name
            message_type: Type of message
            content: Message content
            confidence: Confidence level (0.0-1.0)
            
        Returns:
            Message dictionary
            
        Raises:
            AgentError: If message creation fails
        """
        try:
            if not to_agent or not to_agent.strip():
                raise ValueError("Recipient agent name cannot be empty")
            if not message_type or not message_type.strip():
                raise ValueError("Message type cannot be empty")
            if not (0.0 <= confidence <= 1.0):
                raise ValueError("Confidence must be between 0.0 and 1.0")
            
            message = AgentMessage(
                from_agent=self.agent_name,
                to_agent=to_agent.strip(),
                message_type=message_type.strip(),
                content=content,
                timestamp=time.time(),
                confidence=confidence
            )
            
            self.logger.debug(
                f"Sending {message_type} message to {to_agent} "
                f"(confidence: {confidence:.2f})"
            )
            
            return message.model_dump()
        except Exception as e:
            error_msg = f"Failed to create message: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise AgentError(error_msg, agent_name=self.agent_name) from e
    
    def receive_messages(
        self,
        state: AgentState,
        message_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve messages intended for this agent
        
        Args:
            state: Current agent state
            message_type: Optional filter by message type
            
        Returns:
            List of messages for this agent
        """
        try:
            messages = state.get("agent_messages", [])
            if not messages:
                return []
            
            # Filter messages for this agent
            filtered = [
                msg for msg in messages
                if isinstance(msg, dict) and msg.get("to_agent") == self.agent_name
            ]
            
            # Filter by message type if specified
            if message_type:
                filtered = [
                    msg for msg in filtered
                    if msg.get("message_type") == message_type
                ]
            
            self.logger.debug(
                f"Received {len(filtered)} message(s)"
                + (f" of type {message_type}" if message_type else "")
            )
            
            return filtered
        except Exception as e:
            error_msg = f"Failed to receive messages: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return []
    
    def store_in_memory(self, key: str, value: Any) -> None:
        """
        Store information in agent's local memory
        
        Args:
            key: Memory key
            value: Value to store
        """
        if not key or not key.strip():
            raise ValueError("Memory key cannot be empty")
        
        self.memory.append({
            "key": key.strip(),
            "value": value,
            "timestamp": time.time()
        })
        self.logger.debug(f"Stored in memory: {key}")
    
    def retrieve_from_memory(self, key: str) -> Optional[Any]:
        """
        Retrieve information from agent's local memory
        
        Args:
            key: Memory key
            
        Returns:
            Stored value or None if not found
        """
        if not key or not key.strip():
            raise ValueError("Memory key cannot be empty")
        
        key = key.strip()
        for item in reversed(self.memory):
            if item["key"] == key:
                self.logger.debug(f"Retrieved from memory: {key}")
                return item["value"]
        return None
    
    def log_decision(
        self,
        decision: str,
        reasoning: str,
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """
        Log a decision made by the agent
        
        Args:
            decision: Decision description
            reasoning: Reasoning for the decision
            confidence: Confidence level (0.0-1.0)
            
        Returns:
            Decision dictionary
        """
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        decision_dict = {
            "agent": self.agent_name,
            "decision": decision,
            "reasoning": reasoning,
            "confidence": confidence,
            "timestamp": time.time()
        }
        
        self.logger.info(
            f"Decision: {decision} (confidence: {confidence:.2f})"
        )
        
        return decision_dict
    
    @abstractmethod
    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Process the state and return updates
        
        Args:
            state: Current agent state
            
        Returns:
            Dictionary with state updates
            
        Raises:
            AgentProcessingError: If processing fails
        """
        pass
    
    def _merge_state_updates(
        self,
        state: AgentState,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge state updates with existing state
        
        Args:
            state: Current state
            updates: Updates to merge
            
        Returns:
            Merged state updates
        """
        merged = {}
        
        # Merge agent_messages
        if "agent_messages" in updates:
            current_messages = state.get("agent_messages", [])
            new_message = updates["agent_messages"]
            if isinstance(new_message, list):
                merged["agent_messages"] = current_messages + new_message
            else:
                merged["agent_messages"] = current_messages + [new_message]
        
        # Merge agent_decisions
        if "agent_decisions" in updates:
            current_decisions = state.get("agent_decisions", {})
            new_decision = updates["agent_decisions"]
            if isinstance(new_decision, dict):
                merged["agent_decisions"] = {**current_decisions, **new_decision}
            else:
                merged["agent_decisions"] = {**current_decisions, self.agent_name: new_decision}
        
        # Add other updates
        for key, value in updates.items():
            if key not in ["agent_messages", "agent_decisions"]:
                merged[key] = value
        
        return merged
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.agent_name}, type={self.agent_type})"

