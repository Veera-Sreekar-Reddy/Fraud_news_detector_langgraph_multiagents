"""
Tests for SourceScorerAgent
"""
import pytest
from unittest.mock import patch, Mock
from src.agents.source_scorer import SourceScorerAgent
from src.core.exceptions import AgentProcessingError


class TestSourceScorerAgent:
    """Test suite for SourceScorerAgent"""
    
    def test_initialization(self, mock_config):
        """Test agent initialization"""
        with patch('src.agents.source_scorer.get_config', return_value=mock_config):
            agent = SourceScorerAgent()
            assert agent.agent_name == "SourceScorer"
            assert agent.agent_type == "credibility_assessment"
            assert agent.low_credibility_domains is not None
            assert agent.high_credibility_domains is not None
    
    def test_extract_domain(self, mock_config):
        """Test domain extraction from URL"""
        with patch('src.agents.source_scorer.get_config', return_value=mock_config):
            agent = SourceScorerAgent()
            
            test_cases = [
                ("https://www.example.com/path", "www.example.com"),
                ("http://example.com", "example.com"),
                ("https://subdomain.example.com/page", "subdomain.example.com"),
                ("example.com", "example.com")
            ]
            
            for url, expected_domain in test_cases:
                domain = agent._extract_domain(url)
                assert domain == expected_domain
    
    def test_calculate_credibility_low_credibility_domain(self, mock_config):
        """Test credibility calculation for low credibility domain"""
        with patch('src.agents.source_scorer.get_config', return_value=mock_config):
            agent = SourceScorerAgent()
            
            score, reasoning = agent._calculate_credibility_score("sketchy-site.net")
            assert score < mock_config.low_credibility_threshold
            assert "low-credibility" in reasoning.lower() or "known" in reasoning.lower()
    
    def test_calculate_credibility_high_credibility_domain(self, mock_config):
        """Test credibility calculation for high credibility domain"""
        with patch('src.agents.source_scorer.get_config', return_value=mock_config):
            agent = SourceScorerAgent()
            
            score, reasoning = agent._calculate_credibility_score("reuters.com")
            assert score > mock_config.high_credibility_threshold
            assert "high-credibility" in reasoning.lower() or "known" in reasoning.lower()
    
    def test_calculate_credibility_gov_domain(self, mock_config):
        """Test credibility calculation for .gov domain"""
        with patch('src.agents.source_scorer.get_config', return_value=mock_config):
            agent = SourceScorerAgent()
            
            score, reasoning = agent._calculate_credibility_score("example.gov")
            assert score >= 75
            assert "government" in reasoning.lower()
    
    def test_calculate_credibility_edu_domain(self, mock_config):
        """Test credibility calculation for .edu domain"""
        with patch('src.agents.source_scorer.get_config', return_value=mock_config):
            agent = SourceScorerAgent()
            
            score, reasoning = agent._calculate_credibility_score("example.edu")
            assert score >= 75
            assert "educational" in reasoning.lower()
    
    def test_calculate_credibility_blog_domain(self, mock_config):
        """Test credibility calculation for blog domain"""
        with patch('src.agents.source_scorer.get_config', return_value=mock_config):
            agent = SourceScorerAgent()
            
            score, reasoning = agent._calculate_credibility_score("example.wordpress.com")
            assert score < 50
            assert "blog" in reasoning.lower() or "personal" in reasoning.lower()
    
    def test_calculate_credibility_unknown_domain(self, mock_config):
        """Test credibility calculation for unknown domain"""
        with patch('src.agents.source_scorer.get_config', return_value=mock_config):
            agent = SourceScorerAgent()
            
            score, reasoning = agent._calculate_credibility_score("unknown-site.com")
            assert score == 50
            assert "unknown" in reasoning.lower() or "neutral" in reasoning.lower()
    
    def test_process_low_credibility_source(self, low_credibility_state, mock_config):
        """Test processing a low credibility source"""
        with patch('src.agents.source_scorer.get_config', return_value=mock_config):
            agent = SourceScorerAgent()
            result = agent.process(low_credibility_state)
            
            assert "credibility_score" in result
            assert result["credibility_score"] < mock_config.low_credibility_threshold
            assert "agent_messages" in result
            assert "agent_decisions" in result
    
    def test_process_high_credibility_source(self, high_credibility_state, mock_config):
        """Test processing a high credibility source"""
        with patch('src.agents.source_scorer.get_config', return_value=mock_config):
            agent = SourceScorerAgent()
            result = agent.process(high_credibility_state)
            
            assert "credibility_score" in result
            assert result["credibility_score"] > mock_config.high_credibility_threshold
    
    def test_process_empty_source_url(self, mock_config):
        """Test processing with empty source URL raises error"""
        from src.core.state import create_initial_state
        
        state = create_initial_state(
            query="Test claim",
            source_url=""
        )
        
        with patch('src.agents.source_scorer.get_config', return_value=mock_config):
            agent = SourceScorerAgent()
            with pytest.raises(AgentProcessingError) as exc_info:
                agent.process(state)
            
            assert "empty" in str(exc_info.value).lower()
    
    def test_process_creates_message(self, sample_state, mock_config):
        """Test that process creates a message for Supervisor"""
        with patch('src.agents.source_scorer.get_config', return_value=mock_config):
            agent = SourceScorerAgent()
            result = agent.process(sample_state)
            
            assert len(result["agent_messages"]) > 0
            message = result["agent_messages"][0]
            assert message["to_agent"] == "Supervisor"
            assert message["message_type"] == "credibility_score"
            assert "score" in message["content"]
            assert "domain" in message["content"]
    
    def test_process_logs_decision(self, sample_state, mock_config):
        """Test that process logs a decision"""
        with patch('src.agents.source_scorer.get_config', return_value=mock_config):
            agent = SourceScorerAgent()
            result = agent.process(sample_state)
            
            assert agent.agent_name in result["agent_decisions"]
            decision = result["agent_decisions"][agent.agent_name]
            assert "decision" in decision
            assert "reasoning" in decision
            assert "confidence" in decision
    
    def test_confidence_based_on_score(self, mock_config):
        """Test that confidence is calculated based on credibility score"""
        from src.core.state import create_initial_state
        
        with patch('src.agents.source_scorer.get_config', return_value=mock_config):
            agent = SourceScorerAgent()
            
            # Low credibility should have high confidence
            low_state = create_initial_state(
                query="Test",
                source_url="http://sketchy-site.net"
            )
            result_low = agent.process(low_state)
            decision_low = result_low["agent_decisions"][agent.agent_name]
            assert decision_low["confidence"] >= 0.8
            
            # High credibility should have high confidence
            high_state = create_initial_state(
                query="Test",
                source_url="https://reuters.com"
            )
            result_high = agent.process(high_state)
            decision_high = result_high["agent_decisions"][agent.agent_name]
            assert decision_high["confidence"] >= 0.8
