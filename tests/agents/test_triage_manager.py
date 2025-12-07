"""
Tests for TriageManagerAgent
"""
import pytest
from unittest.mock import patch, Mock
from src.agents.triage_manager import TriageManagerAgent
from src.core.exceptions import AgentProcessingError


class TestTriageManagerAgent:
    """Test suite for TriageManagerAgent"""
    
    def test_initialization(self, mock_config):
        """Test agent initialization"""
        with patch('src.agents.triage_manager.get_config', return_value=mock_config):
            agent = TriageManagerAgent()
            assert agent.agent_name == "TriageManager"
            assert agent.agent_type == "classification"
            assert agent.category_keywords is not None
    
    def test_process_health_claim(self, sample_state, mock_config):
        """Test processing a health-related claim"""
        with patch('src.agents.triage_manager.get_config', return_value=mock_config):
            agent = TriageManagerAgent()
            result = agent.process(sample_state)
            
            assert "category" in result
            assert result["category"] == "health"
            assert "workflow_stage" in result
            assert result["workflow_stage"] == "triaged"
            assert "agent_messages" in result
            assert "agent_decisions" in result
    
    def test_process_finance_claim(self, finance_claim_state, mock_config):
        """Test processing a finance-related claim"""
        with patch('src.agents.triage_manager.get_config', return_value=mock_config):
            agent = TriageManagerAgent()
            result = agent.process(finance_claim_state)
            
            assert result["category"] == "finance"
    
    def test_process_general_claim(self, mock_config):
        """Test processing a general claim with no specific category"""
        from src.core.state import create_initial_state
        
        state = create_initial_state(
            query="This is a general news article.",
            source_url="https://example.com"
        )
        
        with patch('src.agents.triage_manager.get_config', return_value=mock_config):
            agent = TriageManagerAgent()
            result = agent.process(state)
            
            assert result["category"] == "general"
    
    def test_process_empty_query(self, mock_config):
        """Test processing with empty query raises error"""
        from src.core.state import create_initial_state
        
        state = create_initial_state(
            query="",
            source_url="https://example.com"
        )
        
        with patch('src.agents.triage_manager.get_config', return_value=mock_config):
            agent = TriageManagerAgent()
            with pytest.raises(AgentProcessingError) as exc_info:
                agent.process(state)
            
            assert "empty" in str(exc_info.value).lower()
    
    def test_process_creates_message(self, sample_state, mock_config):
        """Test that process creates a message for Supervisor"""
        with patch('src.agents.triage_manager.get_config', return_value=mock_config):
            agent = TriageManagerAgent()
            result = agent.process(sample_state)
            
            assert len(result["agent_messages"]) > 0
            message = result["agent_messages"][0]
            assert message["to_agent"] == "Supervisor"
            assert message["message_type"] == "classification"
            assert "category" in message["content"]
    
    def test_process_logs_decision(self, sample_state, mock_config):
        """Test that process logs a decision"""
        with patch('src.agents.triage_manager.get_config', return_value=mock_config):
            agent = TriageManagerAgent()
            result = agent.process(sample_state)
            
            assert agent.agent_name in result["agent_decisions"]
            decision = result["agent_decisions"][agent.agent_name]
            assert "decision" in decision
            assert "reasoning" in decision
            assert "confidence" in decision
    
    def test_category_keyword_matching(self, mock_config):
        """Test category classification based on keyword matching"""
        test_cases = [
            ("New cancer cure discovered", "health"),
            ("Stock market analysis", "finance"),
            ("Election results announced", "politics"),
            ("Scientific research breakthrough", "science"),
            ("Random news article", "general")
        ]
        
        with patch('src.agents.triage_manager.get_config', return_value=mock_config):
            agent = TriageManagerAgent()
            
            for query, expected_category in test_cases:
                from src.core.state import create_initial_state
                state = create_initial_state(
                    query=query,
                    source_url="https://example.com"
                )
                result = agent.process(state)
                assert result["category"] == expected_category, f"Failed for: {query}"
    
    def test_store_in_memory(self, sample_state, mock_config):
        """Test that category is stored in memory"""
        with patch('src.agents.triage_manager.get_config', return_value=mock_config):
            agent = TriageManagerAgent()
            agent.process(sample_state)
            
            stored_category = agent.retrieve_from_memory("category")
            assert stored_category == "health"
    
    def test_case_insensitive_matching(self, mock_config):
        """Test that keyword matching is case-insensitive"""
        from src.core.state import create_initial_state
        
        state = create_initial_state(
            query="CANCER CURE DISCOVERED TODAY",
            source_url="https://example.com"
        )
        
        with patch('src.agents.triage_manager.get_config', return_value=mock_config):
            agent = TriageManagerAgent()
            result = agent.process(state)
            
            assert result["category"] == "health"
