"""
State management for the multi-agent system
"""
from typing import TypedDict, List, Optional, Dict, Any, Annotated
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import time
from operator import add


class AgentMessage(BaseModel):
    """Message structure for inter-agent communication"""
    from_agent: str = Field(..., description="Sender agent name")
    to_agent: str = Field(..., description="Recipient agent name")
    message_type: str = Field(..., description="Type of message")
    content: Any = Field(..., description="Message content")
    timestamp: float = Field(default_factory=time.time, description="Message timestamp")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence level")
    
    @field_validator('from_agent', 'to_agent', 'message_type')
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "from_agent": "TriageManager",
                "to_agent": "Supervisor",
                "message_type": "classification",
                "content": {"category": "health", "confidence": 0.9},
                "timestamp": 1234567890.0,
                "confidence": 0.9
            }
        }


# Reducer functions for concurrent updates
def add_messages(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add messages from multiple agents"""
    if not isinstance(left, list):
        left = []
    if not isinstance(right, list):
        right = [right] if right else []
    return left + right

def merge_decisions(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge decisions from multiple agents"""
    if not isinstance(left, dict):
        left = {}
    if not isinstance(right, dict):
        right = {right.get("agent", "unknown"): right} if right else {}
    return {**left, **right}

def last_value(left: Any, right: Any) -> Any:
    """Take the rightmost value (latest stage)"""
    return right if right is not None else left

class AgentState(TypedDict, total=False):
    """State structure for the multi-agent system"""
    # Core claim information
    query: str
    source_url: str
    
    # Agent outputs
    category: str
    credibility_score: int
    search_results: List[str]
    fact_check_results: List[Dict[str, Any]]
    sentiment_analysis: Dict[str, Any]
    cross_references: List[Dict[str, Any]]
    logical_fallacies: List[str]
    internal_contradiction: bool
    
    # Agent communication - Use Annotated for concurrent updates
    agent_messages: Annotated[List[Dict[str, Any]], add_messages]
    agent_decisions: Annotated[Dict[str, Any], merge_decisions]
    
    # Workflow state - Use Annotated to handle concurrent updates
    workflow_stage: Annotated[str, last_value]
    requires_review: bool
    review_reason: str
    
    # Final output
    final_verdict: str
    confidence_score: float
    reasoning: str
    evidence_summary: str


def validate_agent_state(state: AgentState) -> bool:
    """Validate agent state structure"""
    required_fields = ["query", "source_url"]
    for field in required_fields:
        if field not in state:
            raise ValueError(f"Missing required field: {field}")
    return True


def create_initial_state(
    query: str,
    source_url: str,
    **kwargs
) -> AgentState:
    """Create initial agent state with default values"""
    state: AgentState = {
        "query": query,
        "source_url": source_url,
        "category": kwargs.get("category", ""),
        "credibility_score": kwargs.get("credibility_score", 0),
        "search_results": kwargs.get("search_results", []),
        "fact_check_results": kwargs.get("fact_check_results", []),
        "sentiment_analysis": kwargs.get("sentiment_analysis", {}),
        "cross_references": kwargs.get("cross_references", []),
        "logical_fallacies": kwargs.get("logical_fallacies", []),
        "internal_contradiction": kwargs.get("internal_contradiction", False),
        "agent_messages": kwargs.get("agent_messages", []),
        "agent_decisions": kwargs.get("agent_decisions", {}),
        "workflow_stage": kwargs.get("workflow_stage", "initialized"),
        "requires_review": kwargs.get("requires_review", False),
        "review_reason": kwargs.get("review_reason", ""),
        "final_verdict": kwargs.get("final_verdict", ""),
        "confidence_score": kwargs.get("confidence_score", 0.0),
        "reasoning": kwargs.get("reasoning", ""),
        "evidence_summary": kwargs.get("evidence_summary", "")
    }
    return state

