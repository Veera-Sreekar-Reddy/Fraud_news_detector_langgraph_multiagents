"""
Workflow Orchestrator
Manages the LangGraph workflow for the multi-agent system
"""
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from ..core.state import AgentState, create_initial_state, validate_agent_state
from ..agents import (
    TriageManagerAgent,
    SourceScorerAgent,
    EvidenceGathererAgent,
    SentimentAnalyzerAgent,
    CrossReferenceAgent,
    LogicalAnalyzerAgent,
    VerdictSynthesizerAgent,
    SupervisorAgent,
    get_agent_registry
)
from ..config import get_config
import logging

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """Orchestrates the multi-agent workflow using LangGraph"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize workflow orchestrator
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = get_config()
        self.registry = get_agent_registry()
        self.app: Optional[StateGraph] = None
        self._build_workflow()
    
    def _build_workflow(self) -> None:
        """Build the LangGraph workflow"""
        logger.info("Building workflow")
        
        # Create agent instances
        triage_manager = TriageManagerAgent()
        source_scorer = SourceScorerAgent()
        evidence_gatherer = EvidenceGathererAgent()
        sentiment_analyzer = SentimentAnalyzerAgent()
        cross_reference = CrossReferenceAgent()
        logical_analyzer = LogicalAnalyzerAgent()
        verdict_synthesizer = VerdictSynthesizerAgent()
        supervisor = SupervisorAgent()
        
        # Register agents
        self.registry.register_agent("TriageManager", triage_manager)
        self.registry.register_agent("SourceScorer", source_scorer)
        self.registry.register_agent("EvidenceGatherer", evidence_gatherer)
        self.registry.register_agent("SentimentAnalyzer", sentiment_analyzer)
        self.registry.register_agent("CrossReferenceAgent", cross_reference)
        self.registry.register_agent("LogicalAnalyzer", logical_analyzer)
        self.registry.register_agent("VerdictSynthesizer", verdict_synthesizer)
        self.registry.register_agent("Supervisor", supervisor)
        
        # Create workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("TriageManager", triage_manager.process)
        workflow.add_node("SourceScorer", source_scorer.process)
        workflow.add_node("EvidenceGatherer", evidence_gatherer.process)
        workflow.add_node("SentimentAnalyzer", sentiment_analyzer.process)
        workflow.add_node("CrossReferenceAgent", cross_reference.process)
        workflow.add_node("LogicalAnalyzer", logical_analyzer.process)
        workflow.add_node("VerdictSynthesizer", verdict_synthesizer.process)
        workflow.add_node("Supervisor", supervisor.process)
        
        # Set entry point
        workflow.set_entry_point("TriageManager")
        
        # After triage, run multiple agents in parallel
        workflow.add_edge("TriageManager", "SourceScorer")
        workflow.add_edge("TriageManager", "EvidenceGatherer")
        workflow.add_edge("TriageManager", "SentimentAnalyzer")
        
        # After evidence gathering, run cross-reference
        workflow.add_edge("EvidenceGatherer", "CrossReferenceAgent")
        
        # Both SourceScorer and CrossReferenceAgent feed into LogicalAnalyzer
        workflow.add_edge("SourceScorer", "LogicalAnalyzer")
        workflow.add_edge("SentimentAnalyzer", "LogicalAnalyzer")
        workflow.add_edge("CrossReferenceAgent", "LogicalAnalyzer")
        
        # Conditional routing based on credibility and contradictions
        workflow.add_conditional_edges(
            "LogicalAnalyzer",
            self._fraud_bypass_router,
            {
                "fast_verdict": "VerdictSynthesizer",
                "full_analysis": "VerdictSynthesizer",
            }
        )
        
        # After verdict synthesis, check if review is needed
        workflow.add_conditional_edges(
            "VerdictSynthesizer",
            self._should_review,
            {
                "review_needed": "Supervisor",
                "final": END,
            }
        )
        
        # Supervisor can either end or trigger additional review
        workflow.add_edge("Supervisor", END)
        
        # Compile the graph
        self.app = workflow.compile()
        logger.info("Workflow built successfully")
    
    def _fraud_bypass_router(self, state: AgentState) -> str:
        """
        Router function for conditional edge based on credibility score
        
        Args:
            state: Current agent state
            
        Returns:
            Route name
        """
        credibility_score = state.get("credibility_score", 50)
        internal_contradiction = state.get("internal_contradiction", False)
        low_threshold = self.config.low_credibility_threshold
        
        if credibility_score < low_threshold and internal_contradiction:
            logger.info("Low credibility + contradictions detected. Routing to FAST VERDICT.")
            return "fast_verdict"
        elif credibility_score < low_threshold:
            logger.info("Low credibility detected. Routing to FAST VERDICT.")
            return "fast_verdict"
        else:
            logger.info("Normal processing. Routing to FULL ANALYSIS.")
            return "full_analysis"
    
    def _should_review(self, state: AgentState) -> str:
        """
        Router to determine if human review is needed
        
        Args:
            state: Current agent state
            
        Returns:
            Route name
        """
        requires_review = state.get("requires_review", False)
        confidence_score = state.get("confidence_score", 0.5)
        
        if requires_review or confidence_score < 0.5:
            logger.info("Review required. Flagging for human review.")
            return "review_needed"
        else:
            logger.info("No review needed. Proceeding to final output.")
            return "final"
    
    def run(self, query: str, source_url: str, **kwargs) -> Dict[str, Any]:
        """
        Run the workflow on a claim
        
        Args:
            query: Claim query
            source_url: Source URL
            **kwargs: Additional state parameters
            
        Returns:
            Final agent state
            
        Raises:
            ValueError: If query or source_url is invalid
        """
        try:
            # Create initial state
            initial_state = create_initial_state(query, source_url, **kwargs)
            
            # Validate state
            validate_agent_state(initial_state)
            
            logger.info(f"Starting workflow for claim: {query[:50]}...")
            
            # Run workflow
            if self.app is None:
                raise RuntimeError("Workflow not initialized")
            
            final_state = self.app.invoke(initial_state)
            
            logger.info("Workflow completed successfully")
            
            return final_state
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}", exc_info=True)
            raise


def create_workflow(config: Optional[Dict[str, Any]] = None) -> WorkflowOrchestrator:
    """
    Create a workflow orchestrator instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Workflow orchestrator instance
    """
    return WorkflowOrchestrator(config=config)

