"""
Agent implementations for the multi-agent system
"""
from .triage_manager import TriageManagerAgent
from .source_scorer import SourceScorerAgent
from .evidence_gatherer import EvidenceGathererAgent
from .sentiment_analyzer import SentimentAnalyzerAgent
from .cross_reference import CrossReferenceAgent
from .logical_analyzer import LogicalAnalyzerAgent
from .verdict_synthesizer import VerdictSynthesizerAgent
from .supervisor import SupervisorAgent
from .registry import AgentRegistry, get_agent_registry

__all__ = [
    "TriageManagerAgent",
    "SourceScorerAgent",
    "EvidenceGathererAgent",
    "SentimentAnalyzerAgent",
    "CrossReferenceAgent",
    "LogicalAnalyzerAgent",
    "VerdictSynthesizerAgent",
    "SupervisorAgent",
    "AgentRegistry",
    "get_agent_registry"
]

