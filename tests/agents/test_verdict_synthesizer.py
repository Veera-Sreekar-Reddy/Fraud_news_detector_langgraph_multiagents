"""
Tests for VerdictSynthesizerAgent
"""
import pytest
from unittest.mock import patch, Mock
from src.agents.verdict_synthesizer import VerdictSynthesizerAgent
from src.core.exceptions import AgentProcessingError


class TestVerdictSynthesizerAgent:
    """Test suite for VerdictSynthesizerAgent"""
    
    def test_initialization_with_llama3(self, mock_config, mock_llama3_client):
        """Test agent initialization with Llama3 API"""
        with patch('src.agents.verdict_synthesizer.get_config', return_value=mock_config), \
             patch('src.agents.verdict_synthesizer.get_llama3_client', return_value=mock_llama3_client):
            
            agent = VerdictSynthesizerAgent()
            assert agent.agent_name == "VerdictSynthesizer"
            assert agent.use_llama3 is True
    
    def test_initialization_without_llama3(self, mock_config):
        """Test agent initialization without Llama3 API"""
        with patch('src.agents.verdict_synthesizer.get_config', return_value=mock_config), \
             patch('src.agents.verdict_synthesizer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = VerdictSynthesizerAgent()
            assert agent.use_llama3 is False
    
    def test_synthesize_verdict_high_confidence_false(self, state_with_fallacies, mock_config):
        """Test verdict synthesis for high confidence FALSE"""
        with patch('src.agents.verdict_synthesizer.get_config', return_value=mock_config), \
             patch('src.agents.verdict_synthesizer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = VerdictSynthesizerAgent()
            state_with_fallacies["credibility_score"] = 15
            state_with_fallacies["internal_contradiction"] = True
            
            result = agent._synthesize_verdict(state_with_fallacies)
            
            assert "FALSE" in result["verdict"].upper()
            assert result["confidence_score"] >= mock_config.high_confidence_threshold
    
    def test_synthesize_verdict_medium_confidence_false(self, state_with_evidence, mock_config):
        """Test verdict synthesis for medium confidence FALSE"""
        with patch('src.agents.verdict_synthesizer.get_config', return_value=mock_config), \
             patch('src.agents.verdict_synthesizer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = VerdictSynthesizerAgent()
            state_with_evidence["fact_check_results"] = [
                {"verdict": "False", "confidence": 0.90}
            ]
            
            result = agent._synthesize_verdict(state_with_evidence)
            
            assert "FALSE" in result["verdict"].upper()
            assert result["confidence_score"] >= mock_config.medium_confidence_threshold
    
    def test_synthesize_verdict_misleading(self, state_with_sentiment, mock_config):
        """Test verdict synthesis for MISLEADING"""
        with patch('src.agents.verdict_synthesizer.get_config', return_value=mock_config), \
             patch('src.agents.verdict_synthesizer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = VerdictSynthesizerAgent()
            state_with_sentiment["logical_fallacies"] = ["false_dilemma"]
            state_with_sentiment["search_results"] = ["Some evidence"]
            
            result = agent._synthesize_verdict(state_with_sentiment)
            
            assert "MISLEADING" in result["verdict"].upper()
            assert result["confidence_score"] >= mock_config.low_confidence_threshold
    
    def test_synthesize_verdict_unverifiable(self, sample_state, mock_config):
        """Test verdict synthesis for UNVERIFIABLE"""
        with patch('src.agents.verdict_synthesizer.get_config', return_value=mock_config), \
             patch('src.agents.verdict_synthesizer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = VerdictSynthesizerAgent()
            result = agent._synthesize_verdict(sample_state)
            
            assert "UNVERIFIABLE" in result["verdict"].upper()
            assert result["confidence_score"] < 0.5
    
    def test_synthesize_verdict_with_llama3(self, state_complete, mock_config, mock_llama3_client):
        """Test verdict synthesis using Llama3"""
        with patch('src.agents.verdict_synthesizer.get_config', return_value=mock_config), \
             patch('src.agents.verdict_synthesizer.get_llama3_client', return_value=mock_llama3_client):
            
            mock_llama3_client.analyze_claim.return_value = {
                "verdict": "FALSE",
                "confidence": 0.85,
                "reasoning": "Test reasoning",
                "evidence": "Test evidence"
            }
            
            agent = VerdictSynthesizerAgent()
            result = agent._synthesize_verdict(state_complete)
            
            assert "verdict" in result
            assert "confidence_score" in result
            assert "reasoning" in result
            mock_llama3_client.analyze_claim.assert_called_once()
    
    def test_synthesize_verdict_requires_review(self, sample_state, mock_config):
        """Test that verdict synthesis flags review when needed"""
        with patch('src.agents.verdict_synthesizer.get_config', return_value=mock_config), \
             patch('src.agents.verdict_synthesizer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = VerdictSynthesizerAgent()
            sample_state["logical_fallacies"] = ["fallacy1", "fallacy2", "fallacy3"]
            
            result = agent._synthesize_verdict(sample_state)
            
            assert result["requires_review"] is True
            assert "review_reason" in result
    
    def test_process_creates_final_verdict(self, state_complete, mock_config):
        """Test that process creates final verdict"""
        with patch('src.agents.verdict_synthesizer.get_config', return_value=mock_config), \
             patch('src.agents.verdict_synthesizer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = VerdictSynthesizerAgent()
            result = agent.process(state_complete)
            
            assert "final_verdict" in result
            assert "confidence_score" in result
            assert "reasoning" in result
            assert "evidence_summary" in result
            assert "workflow_stage" in result
            assert result["workflow_stage"] == "completed"
    
    def test_process_logs_decision(self, state_complete, mock_config):
        """Test that process logs a decision"""
        with patch('src.agents.verdict_synthesizer.get_config', return_value=mock_config), \
             patch('src.agents.verdict_synthesizer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = VerdictSynthesizerAgent()
            result = agent.process(state_complete)
            
            assert agent.agent_name in result["agent_decisions"]
            decision = result["agent_decisions"][agent.agent_name]
            assert "decision" in decision
            assert "reasoning" in decision
            assert "confidence" in decision
    
    def test_verdict_prioritization(self, mock_config):
        """Test that verdict prioritization works correctly"""
        from src.core.state import create_initial_state
        
        with patch('src.agents.verdict_synthesizer.get_config', return_value=mock_config), \
             patch('src.agents.verdict_synthesizer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = VerdictSynthesizerAgent()
            
            # High confidence FALSE should take priority
            state = create_initial_state(
                query="Test",
                source_url="http://sketchy-site.net"
            )
            state["credibility_score"] = 15
            state["internal_contradiction"] = True
            state["fact_check_results"] = [{"verdict": "False"}]
            
            result = agent._synthesize_verdict(state)
            assert "FALSE" in result["verdict"].upper()
            assert result["confidence_score"] >= mock_config.high_confidence_threshold
