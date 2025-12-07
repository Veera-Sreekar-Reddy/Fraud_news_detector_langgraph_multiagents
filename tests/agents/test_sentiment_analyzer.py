"""
Tests for SentimentAnalyzerAgent
"""
import pytest
from unittest.mock import patch, Mock
from src.agents.sentiment_analyzer import SentimentAnalyzerAgent
from src.core.exceptions import AgentProcessingError


class TestSentimentAnalyzerAgent:
    """Test suite for SentimentAnalyzerAgent"""
    
    def test_initialization_with_llama3(self, mock_config, mock_llama3_client):
        """Test agent initialization with Llama3 API"""
        with patch('src.agents.sentiment_analyzer.get_config', return_value=mock_config), \
             patch('src.agents.sentiment_analyzer.get_llama3_client', return_value=mock_llama3_client):
            
            agent = SentimentAnalyzerAgent()
            assert agent.agent_name == "SentimentAnalyzer"
            assert agent.use_llama3 is True
    
    def test_initialization_without_llama3(self, mock_config):
        """Test agent initialization without Llama3 API"""
        with patch('src.agents.sentiment_analyzer.get_config', return_value=mock_config), \
             patch('src.agents.sentiment_analyzer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = SentimentAnalyzerAgent()
            assert agent.use_llama3 is False
    
    def test_analyze_sentiment_with_llama3(self, mock_config, mock_llama3_client, mock_llama3_sentiment_response):
        """Test sentiment analysis using Llama3"""
        with patch('src.agents.sentiment_analyzer.get_config', return_value=mock_config), \
             patch('src.agents.sentiment_analyzer.get_llama3_client', return_value=mock_llama3_client):
            
            mock_llama3_client.analyze_claim.return_value = mock_llama3_sentiment_response
            
            agent = SentimentAnalyzerAgent()
            result = agent._analyze_sentiment("Shocking secret they don't want you to know!")
            
            assert "sentiment" in result
            assert result["sentiment"] in ["positive", "negative", "neutral"]
            assert "manipulation_score" in result
            assert "is_emotional_appeal" in result
            mock_llama3_client.analyze_claim.assert_called_once()
    
    def test_analyze_sentiment_fallback(self, mock_config):
        """Test sentiment analysis using fallback rule-based method"""
        with patch('src.agents.sentiment_analyzer.get_config', return_value=mock_config), \
             patch('src.agents.sentiment_analyzer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = SentimentAnalyzerAgent()
            result = agent._analyze_sentiment("Shocking secret they don't want you to know!")
            
            assert "sentiment" in result
            assert result["sentiment"] in ["positive", "negative", "neutral"]
            assert "manipulation_score" in result
            assert "manipulative_phrases_found" in result
    
    def test_analyze_sentiment_positive(self, mock_config):
        """Test sentiment analysis for positive sentiment"""
        with patch('src.agents.sentiment_analyzer.get_config', return_value=mock_config), \
             patch('src.agents.sentiment_analyzer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = SentimentAnalyzerAgent()
            result = agent._analyze_sentiment("Great cure discovered for amazing breakthrough")
            
            assert result["sentiment"] == "positive"
    
    def test_analyze_sentiment_negative(self, mock_config):
        """Test sentiment analysis for negative sentiment"""
        with patch('src.agents.sentiment_analyzer.get_config', return_value=mock_config), \
             patch('src.agents.sentiment_analyzer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = SentimentAnalyzerAgent()
            result = agent._analyze_sentiment("Danger warning threat scam fake")
            
            assert result["sentiment"] == "negative"
    
    def test_analyze_sentiment_manipulative(self, mock_config):
        """Test sentiment analysis detects manipulative language"""
        with patch('src.agents.sentiment_analyzer.get_config', return_value=mock_config), \
             patch('src.agents.sentiment_analyzer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = SentimentAnalyzerAgent()
            result = agent._analyze_sentiment("Shocking secret they don't want you to know conspiracy")
            
            assert result["manipulation_score"] > 0.5
            assert result["is_emotional_appeal"] is True
            assert result["manipulative_phrases_found"] > 0
    
    def test_process_creates_sentiment_analysis(self, sample_state, mock_config):
        """Test that process creates sentiment analysis"""
        with patch('src.agents.sentiment_analyzer.get_config', return_value=mock_config), \
             patch('src.agents.sentiment_analyzer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = SentimentAnalyzerAgent()
            result = agent.process(sample_state)
            
            assert "sentiment_analysis" in result
            assert "sentiment" in result["sentiment_analysis"]
            assert "manipulation_score" in result["sentiment_analysis"]
    
    def test_process_empty_query(self, mock_config):
        """Test processing with empty query raises error"""
        from src.core.state import create_initial_state
        
        state = create_initial_state(
            query="",
            source_url="https://example.com"
        )
        
        with patch('src.agents.sentiment_analyzer.get_config', return_value=mock_config), \
             patch('src.agents.sentiment_analyzer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = SentimentAnalyzerAgent()
            with pytest.raises(AgentProcessingError) as exc_info:
                agent.process(state)
            
            assert "empty" in str(exc_info.value).lower()
    
    def test_process_creates_message(self, sample_state, mock_config):
        """Test that process creates a message for LogicalAnalyzer"""
        with patch('src.agents.sentiment_analyzer.get_config', return_value=mock_config), \
             patch('src.agents.sentiment_analyzer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = SentimentAnalyzerAgent()
            result = agent.process(sample_state)
            
            assert len(result["agent_messages"]) > 0
            message = result["agent_messages"][0]
            assert message["to_agent"] == "LogicalAnalyzer"
            assert message["message_type"] == "sentiment"
            assert "sentiment" in message["content"]
    
    def test_process_logs_decision(self, sample_state, mock_config):
        """Test that process logs a decision"""
        with patch('src.agents.sentiment_analyzer.get_config', return_value=mock_config), \
             patch('src.agents.sentiment_analyzer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = SentimentAnalyzerAgent()
            result = agent.process(sample_state)
            
            assert agent.agent_name in result["agent_decisions"]
            decision = result["agent_decisions"][agent.agent_name]
            assert "decision" in decision
            assert "reasoning" in decision
    
    def test_manipulation_score_calculation(self, mock_config):
        """Test manipulation score calculation"""
        with patch('src.agents.sentiment_analyzer.get_config', return_value=mock_config), \
             patch('src.agents.sentiment_analyzer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = SentimentAnalyzerAgent()
            
            # Test with multiple manipulative phrases
            result = agent._analyze_sentiment("Shocking secret conspiracy cover-up")
            assert result["manipulation_score"] > 0.5
            
            # Test with no manipulative phrases
            result = agent._analyze_sentiment("Normal news article about events")
            assert result["manipulation_score"] < 0.5
