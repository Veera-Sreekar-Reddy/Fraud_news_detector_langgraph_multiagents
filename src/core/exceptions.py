"""
Custom exceptions for the multi-agent system
"""
from typing import Optional


class AgentError(Exception):
    """Base exception for agent-related errors"""
    def __init__(self, message: str, agent_name: Optional[str] = None):
        self.agent_name = agent_name
        super().__init__(message)


class StateError(Exception):
    """Exception for state-related errors"""
    def __init__(self, message: str, state_key: Optional[str] = None):
        self.state_key = state_key
        super().__init__(message)


class ValidationError(Exception):
    """Exception for validation errors"""
    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        super().__init__(message)


class AgentProcessingError(AgentError):
    """Exception raised when an agent fails to process state"""
    pass


class AgentCommunicationError(AgentError):
    """Exception raised when agent communication fails"""
    pass

