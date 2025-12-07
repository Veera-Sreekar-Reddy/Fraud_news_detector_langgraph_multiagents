"""
Tests for CrossReferenceAgent
"""
import pytest
from unittest.mock import patch
from src.agents.cross_reference import CrossReferenceAgent
from src.core.exceptions import AgentProcessingError


class TestCrossReferenceAgent:
    """Test suite for CrossReferenceAgent"""
    
    def test_initialization(self, mock_config):
        """Test agent initialization"""
        with patch('src.agents.cross_reference.get_config', return_value=mock_config):
            agent = CrossReferenceAgent()
            assert agent.agent_name == "CrossReferenceAgent"
            assert agent.agent_type == "cross_referencing"
    
    def test_analyze_consensus_no_results(self, mock_config):
        """Test consensus analysis with no fact-check results"""
        with patch('src.agents.cross_reference.get_config', return_value=mock_config):
            agent = CrossReferenceAgent()
            result = agent._analyze_consensus([])
            
            assert result["consensus"] is False
            assert result["consensus_level"] == "none"
            assert len(result["verdicts"]) == 0
    
    def test_analyze_consensus_single_result(self, mock_config):
        """Test consensus analysis with single result"""
        with patch('src.agents.cross_reference.get_config', return_value=mock_config):
            agent = CrossReferenceAgent()
            fact_checks = [
                {"verdict": "FALSE", "confidence": 0.90}
            ]
            result = agent._analyze_consensus(fact_checks)
            
            assert result["consensus"] is True
            assert result["consensus_level"] == "low"
            assert len(result["verdicts"]) == 1
    
    def test_analyze_consensus_multiple_same_verdict(self, mock_config, mock_fact_check_results):
        """Test consensus analysis with multiple results having same verdict"""
        with patch('src.agents.cross_reference.get_config', return_value=mock_config):
            agent = CrossReferenceAgent()
            result = agent._analyze_consensus(mock_fact_check_results)
            
            assert result["consensus"] is True
            assert result["consensus_level"] in ["medium", "high"]
            assert len(result["verdicts"]) == 2
    
    def test_analyze_consensus_conflicting_verdicts(self, mock_config):
        """Test consensus analysis with conflicting verdicts"""
        with patch('src.agents.cross_reference.get_config', return_value=mock_config):
            agent = CrossReferenceAgent()
            fact_checks = [
                {"verdict": "FALSE", "confidence": 0.90},
                {"verdict": "TRUE", "confidence": 0.85}
            ]
            result = agent._analyze_consensus(fact_checks)
            
            assert result["consensus"] is False
            assert len(result["unique_verdicts"]) == 2
    
    def test_create_cross_references(self, mock_config, mock_fact_check_results):
        """Test creating cross-reference entries"""
        with patch('src.agents.cross_reference.get_config', return_value=mock_config):
            agent = CrossReferenceAgent()
            result = agent._create_cross_references(mock_fact_check_results)
            
            assert len(result) == len(mock_fact_check_results)
            assert all("source" in ref for ref in result)
            assert all("verdict" in ref for ref in result)
            assert all("confidence" in ref for ref in result)
    
    def test_create_cross_references_matches_flag(self, mock_config):
        """Test that matches_other_sources flag is set correctly"""
        with patch('src.agents.cross_reference.get_config', return_value=mock_config):
            agent = CrossReferenceAgent()
            
            # Single result
            single_result = agent._create_cross_references([{"verdict": "FALSE"}])
            assert single_result[0]["matches_other_sources"] is False
            
            # Multiple results
            multiple_results = agent._create_cross_references([
                {"verdict": "FALSE"},
                {"verdict": "FALSE"}
            ])
            assert multiple_results[0]["matches_other_sources"] is True
    
    def test_process_with_fact_check_results(self, state_with_evidence, mock_config):
        """Test processing with fact-check results"""
        with patch('src.agents.cross_reference.get_config', return_value=mock_config):
            agent = CrossReferenceAgent()
            result = agent.process(state_with_evidence)
            
            assert "cross_references" in result
            assert len(result["cross_references"]) > 0
            assert "agent_messages" in result
            assert "agent_decisions" in result
    
    def test_process_without_fact_check_results(self, sample_state, mock_config):
        """Test processing without fact-check results"""
        with patch('src.agents.cross_reference.get_config', return_value=mock_config):
            agent = CrossReferenceAgent()
            result = agent.process(sample_state)
            
            assert "cross_references" in result
            assert isinstance(result["cross_references"], list)
    
    def test_process_creates_message(self, state_with_evidence, mock_config):
        """Test that process creates a message for LogicalAnalyzer"""
        with patch('src.agents.cross_reference.get_config', return_value=mock_config):
            agent = CrossReferenceAgent()
            result = agent.process(state_with_evidence)
            
            assert len(result["agent_messages"]) > 0
            message = result["agent_messages"][0]
            assert message["to_agent"] == "LogicalAnalyzer"
            assert message["message_type"] == "cross_reference"
            assert "cross_references" in message["content"]
            assert "consensus" in message["content"]
    
    def test_process_logs_decision(self, state_with_evidence, mock_config):
        """Test that process logs a decision"""
        with patch('src.agents.cross_reference.get_config', return_value=mock_config):
            agent = CrossReferenceAgent()
            result = agent.process(state_with_evidence)
            
            assert agent.agent_name in result["agent_decisions"]
            decision = result["agent_decisions"][agent.agent_name]
            assert "decision" in decision
            assert "reasoning" in decision
            assert "consensus" in decision["reasoning"].lower()
    
    def test_consensus_level_calculation(self, mock_config):
        """Test consensus level calculation"""
        with patch('src.agents.cross_reference.get_config', return_value=mock_config):
            agent = CrossReferenceAgent()
            
            # Low consensus (1 result)
            result = agent._analyze_consensus([{"verdict": "FALSE"}])
            assert result["consensus_level"] == "low"
            
            # Medium consensus (2 results)
            result = agent._analyze_consensus([
                {"verdict": "FALSE"},
                {"verdict": "FALSE"}
            ])
            assert result["consensus_level"] == "medium"
            
            # High consensus (3+ results)
            result = agent._analyze_consensus([
                {"verdict": "FALSE"},
                {"verdict": "FALSE"},
                {"verdict": "FALSE"}
            ])
            assert result["consensus_level"] == "high"
