"""
Verdict Synthesizer Agent
Responsible for synthesizing final verdict
"""
from typing import Dict, Any, List
from ..core.base_agent import BaseAgent
from ..core.state import AgentState
from ..core.exceptions import AgentProcessingError
from ..config import get_config
from ..integrations import get_llama3_client


class VerdictSynthesizerAgent(BaseAgent):
    """Agent responsible for synthesizing final verdict"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Verdict Synthesizer Agent
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__("VerdictSynthesizer", "verdict_synthesis")
        self.config = get_config()
        self.low_threshold = self.config.low_credibility_threshold
        self.high_confidence = self.config.high_confidence_threshold
        self.medium_confidence = self.config.medium_confidence_threshold
        self.low_confidence = self.config.low_confidence_threshold
        
        try:
            self.llama3_client = get_llama3_client()
            self.use_llama3 = True
        except Exception as e:
            self.logger.warning(f"Llama3 API not available, using fallback: {e}")
            self.llama3_client = None
            self.use_llama3 = False
    
    def _synthesize_verdict(self, state: AgentState) -> Dict[str, Any]:
        """
        Synthesize final verdict from all evidence using Llama3.3 70B
        
        Args:
            state: Current agent state
            
        Returns:
            Verdict dictionary
        """
        credibility_score = state.get("credibility_score", 50)
        internal_contradiction = state.get("internal_contradiction", False)
        fact_check_results = state.get("fact_check_results", [])
        search_results = state.get("search_results", [])
        logical_fallacies = state.get("logical_fallacies", [])
        sentiment_analysis = state.get("sentiment_analysis", {})
        query = state.get("query", "")
        
        # Use Llama3 for final verdict synthesis if available
        use_llama3_result = False
        
        if self.use_llama3 and self.llama3_client and query:
            try:
                self.logger.info("Using Llama3.3 70B for verdict synthesis")
                
                # Build comprehensive context for Llama3
                fact_check_summary = ""
                if fact_check_results:
                    verdicts = [fc.get('verdict', 'Unknown') for fc in fact_check_results]
                    fact_check_summary = f"Fact-check verdicts: {', '.join(set(verdicts))}"
                
                context = f"""Synthesize a final verdict based on this comprehensive analysis:

Source Credibility: {credibility_score}/100 ({'Low' if credibility_score < 20 else 'High' if credibility_score > 80 else 'Medium'})
Internal Contradictions: {'Yes' if internal_contradiction else 'No'}
Fact-Check Results: {len(fact_check_results)} sources analyzed
{fact_check_summary}
Logical Fallacies Detected: {len(logical_fallacies)} ({', '.join(logical_fallacies[:3]) if logical_fallacies else 'None'})
Sentiment: {sentiment_analysis.get('sentiment', 'unknown')} (Manipulation: {sentiment_analysis.get('manipulation_score', 0.0):.2f})
Evidence Found: {len(search_results)} items

Based on these indicators, determine if the claim is:
- TRUE: If credible source and evidence supports it
- FALSE: If low credibility, contradictions, or false fact-checks
- MISLEADING: If partial truth with distortions or fallacies
- UNVERIFIABLE: Only if truly cannot determine (use sparingly)

Analyze based on patterns, source credibility, logical consistency, and evidence quality."""
                
                analysis = self.llama3_client.analyze_claim(
                    claim=query,
                    context=context,
                    analysis_type="fact_check"
                )
                
                if isinstance(analysis, dict) and "verdict" in analysis:
                    # Use Llama3's verdict
                    llama_verdict_raw = analysis.get("verdict", "UNVERIFIABLE").upper().strip()
                    llama_confidence = float(analysis.get("confidence", 0.5))
                    llama_reasoning = analysis.get("reasoning", "")
                    
                    # Map Llama3 verdict to our format
                    verdict_map = {
                        "TRUE": "TRUE",
                        "FALSE": "FALSE", 
                        "MISLEADING": "MISLEADING",
                        "UNVERIFIABLE": "UNVERIFIABLE"
                    }
                    llama_verdict = verdict_map.get(llama_verdict_raw, "UNVERIFIABLE")
                    
                    # Check if Llama3 is being too cautious (saying can't verify)
                    is_too_cautious = (
                        llama_verdict == "UNVERIFIABLE" and 
                        llama_reasoning and 
                        ("cannot" in llama_reasoning.lower() or 
                         "unable" in llama_reasoning.lower() or
                         "don't have" in llama_reasoning.lower() or
                         "not have access" in llama_reasoning.lower() or
                         "impossible to verify" in llama_reasoning.lower())
                    )
                    
                    # If Llama3 is being too cautious and we have other evidence, use rule-based logic
                    if is_too_cautious and (fact_check_results or credibility_score < 20 or credibility_score > 80):
                        self.logger.warning("Llama3 returned UNVERIFIABLE due to lack of real-time info, using rule-based analysis instead")
                        # Fall through to rule-based logic
                    elif llama_verdict != "UNVERIFIABLE" or (llama_reasoning and not is_too_cautious):
                        # Use Llama3's verdict if it's not UNVERIFIABLE or if it has good reasoning
                        verdict = llama_verdict
                        confidence_score = llama_confidence
                        reasoning_parts = [llama_reasoning] if llama_reasoning else []
                        evidence_summary = [analysis.get("evidence", "")] if analysis.get("evidence") else []
                        use_llama3_result = True
                    else:
                        # Llama3 returned unverifiable - fall through to rule-based
                        self.logger.warning("Llama3 returned UNVERIFIABLE, using rule-based fallback")
                else:
                    # Fall through to rule-based logic
                    self.logger.warning("Llama3 response format unexpected, using fallback")
            except Exception as e:
                self.logger.warning(f"Llama3 API call failed, using fallback: {e}")
                # Fall through to rule-based logic
        
        # Fallback to rule-based synthesis if Llama3 not used or failed
        if not use_llama3_result:
            verdict = "UNVERIFIABLE"
            confidence_score = 0.5
            reasoning_parts = []
            evidence_summary = []
            
            # High confidence FALSE
            if credibility_score < self.low_threshold and internal_contradiction:
                verdict = "FALSE (High Confidence - Known Disinformation Source)"
                confidence_score = self.high_confidence
                reasoning_parts.append(f"Low credibility source (< {self.low_threshold})")
                reasoning_parts.append("Internal contradictions detected")
                evidence_summary.append(f"Credibility score: {credibility_score}")
                evidence_summary.append("Multiple fact-checkers contradict claim")
            # Medium confidence FALSE
            elif fact_check_results and any(r.get("verdict") == "False" for r in fact_check_results):
                verdict = "FALSE (Medium Confidence - Fact-Checked)"
                confidence_score = self.medium_confidence
                reasoning_parts.append("Fact-checkers rate as false")
                reasoning_parts.append(f"Found {len(search_results)} contradicting evidence")
                evidence_summary.extend([
                    f"{r.get('source')}: {r.get('verdict')}"
                    for r in fact_check_results
                ])
            # MISLEADING
            elif search_results or logical_fallacies:
                verdict = "MISLEADING (Low Confidence - Contradictory Evidence)"
                confidence_score = self.low_confidence
                reasoning_parts.append("Contradictory evidence found")
                if logical_fallacies:
                    reasoning_parts.append(f"Detected {len(logical_fallacies)} logical fallacies")
                if sentiment_analysis.get("is_emotional_appeal"):
                    reasoning_parts.append("Emotional manipulation detected")
                evidence_summary.extend(search_results[:3])
            # UNVERIFIABLE
            else:
                verdict = "UNVERIFIABLE (Insufficient Evidence)"
                confidence_score = 0.4
                reasoning_parts.append("Insufficient evidence to make determination")
                evidence_summary.append("No fact-check results available")
        
        # Check if review is needed
        requires_review = False
        review_reason = ""
        
        if confidence_score < 0.6 and not fact_check_results:
            requires_review = True
            review_reason = "Low confidence and no fact-check results"
        elif len(logical_fallacies) > 2:
            requires_review = True
            review_reason = "Multiple logical fallacies detected"
        
        reasoning = ". ".join(reasoning_parts)
        evidence_summary_text = "\n".join(evidence_summary) if evidence_summary else "No evidence collected"
        
        return {
            "verdict": verdict,
            "confidence_score": confidence_score,
            "reasoning": reasoning,
            "evidence_summary": evidence_summary_text,
            "requires_review": requires_review,
            "review_reason": review_reason
        }
    
    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Process state and synthesize verdict
        
        Args:
            state: Current agent state
            
        Returns:
            Dictionary with state updates
            
        Raises:
            AgentProcessingError: If processing fails
        """
        try:
            self.logger.info("Synthesizing final verdict")
            
            # Synthesize verdict
            verdict_data = self._synthesize_verdict(state)
            
            self.logger.info(
                f"Verdict: {verdict_data['verdict']} "
                f"(confidence: {verdict_data['confidence_score']:.2%})"
            )
            
            if verdict_data["requires_review"]:
                self.logger.warning(f"Review required: {verdict_data['review_reason']}")
            
            decision = self.log_decision(
                decision=verdict_data["verdict"],
                reasoning=verdict_data["reasoning"],
                confidence=verdict_data["confidence_score"]
            )
            
            return {
                "final_verdict": verdict_data["verdict"],
                "confidence_score": verdict_data["confidence_score"],
                "reasoning": verdict_data["reasoning"],
                "evidence_summary": verdict_data["evidence_summary"],
                "requires_review": verdict_data["requires_review"],
                "review_reason": verdict_data["review_reason"],
                "workflow_stage": "completed",
                "agent_decisions": {self.agent_name: decision}  # Reducer will merge
            }
        except Exception as e:
            error_msg = f"Failed to synthesize verdict: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise AgentProcessingError(error_msg, agent_name=self.agent_name) from e

