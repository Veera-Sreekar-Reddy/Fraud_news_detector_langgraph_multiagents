"""
Tests for EvidenceGathererAgent
"""
import pytest
from unittest.mock import patch, Mock, MagicMock
from src.agents.evidence_gatherer import EvidenceGathererAgent
from src.core.exceptions import AgentProcessingError


class TestEvidenceGathererAgent:
    """Test suite for EvidenceGathererAgent"""
    
    def test_initialization_with_apis(self, mock_config, mock_google_fact_check_client, mock_llama3_client):
        """Test agent initialization with both APIs available"""
        with patch('src.agents.evidence_gatherer.get_config', return_value=mock_config), \
             patch('src.agents.evidence_gatherer.get_google_fact_check_client', return_value=mock_google_fact_check_client), \
             patch('src.agents.evidence_gatherer.get_llama3_client', return_value=mock_llama3_client):
            
            agent = EvidenceGathererAgent()
            assert agent.agent_name == "EvidenceGatherer"
            assert agent.use_google_fact_check is True
            assert agent.use_llama3 is True
    
    def test_initialization_without_google_api(self, mock_config, mock_llama3_client):
        """Test agent initialization without Google Fact Check API"""
        with patch('src.agents.evidence_gatherer.get_config', return_value=mock_config), \
             patch('src.agents.evidence_gatherer.get_google_fact_check_client', side_effect=Exception("API not available")), \
             patch('src.agents.evidence_gatherer.get_llama3_client', return_value=mock_llama3_client):
            
            agent = EvidenceGathererAgent()
            assert agent.use_google_fact_check is False
            assert agent.use_llama3 is True
    
    def test_initialization_without_llama3_api(self, mock_config, mock_google_fact_check_client):
        """Test agent initialization without Llama3 API"""
        with patch('src.agents.evidence_gatherer.get_config', return_value=mock_config), \
             patch('src.agents.evidence_gatherer.get_google_fact_check_client', return_value=mock_google_fact_check_client), \
             patch('src.agents.evidence_gatherer.get_llama3_client', side_effect=Exception("API not available")):
            
            agent = EvidenceGathererAgent()
            assert agent.use_google_fact_check is True
            assert agent.use_llama3 is False
    
    def test_gather_evidence_with_google_api(self, mock_config, mock_google_fact_check_client):
        """Test evidence gathering using Google Fact Check API"""
        from src.core.state import create_initial_state
        
        state = create_initial_state(
            query="Test claim",
            source_url="https://example.com"
        )
        state["category"] = "health"
        
        with patch('src.agents.evidence_gatherer.get_config', return_value=mock_config), \
             patch('src.agents.evidence_gatherer.get_google_fact_check_client', return_value=mock_google_fact_check_client), \
             patch('src.agents.evidence_gatherer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = EvidenceGathererAgent()
            search_results, fact_check_results = agent._gather_evidence_by_category("health", "Test claim")
            
            assert len(fact_check_results) > 0
            assert fact_check_results[0]["source"] == "Snopes"
            mock_google_fact_check_client.check_claim.assert_called_once()
    
    def test_gather_evidence_with_llama3_fallback(self, mock_config, mock_llama3_client):
        """Test evidence gathering using Llama3 as fallback"""
        from src.core.state import create_initial_state
        
        # Mock Google API to return no results
        mock_google = Mock()
        mock_google.check_claim.return_value = {
            "found": False,
            "verdict": "UNVERIFIABLE",
            "confidence": 0.40,
            "reasoning": "No results found",
            "results": []
        }
        
        with patch('src.agents.evidence_gatherer.get_config', return_value=mock_config), \
             patch('src.agents.evidence_gatherer.get_google_fact_check_client', return_value=mock_google), \
             patch('src.agents.evidence_gatherer.get_llama3_client', return_value=mock_llama3_client):
            
            agent = EvidenceGathererAgent()
            search_results, fact_check_results = agent._gather_evidence_by_category("health", "Test claim")
            
            # Should have Llama3 results
            assert len(fact_check_results) > 0
            mock_llama3_client.analyze_claim.assert_called_once()
    
    def test_gather_evidence_fallback_simulation(self, mock_config):
        """Test evidence gathering with fallback simulation"""
        with patch('src.agents.evidence_gatherer.get_config', return_value=mock_config), \
             patch('src.agents.evidence_gatherer.get_google_fact_check_client', side_effect=Exception("Not available")), \
             patch('src.agents.evidence_gatherer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = EvidenceGathererAgent()
            search_results, fact_check_results = agent._gather_evidence_by_category("health", "Test claim")
            
            # Should have fallback results
            assert len(search_results) > 0
            assert len(fact_check_results) > 0
    
    def test_process_health_category(self, health_claim_state, mock_config, mock_google_fact_check_client):
        """Test processing a health category claim"""
        with patch('src.agents.evidence_gatherer.get_config', return_value=mock_config), \
             patch('src.agents.evidence_gatherer.get_google_fact_check_client', return_value=mock_google_fact_check_client), \
             patch('src.agents.evidence_gatherer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = EvidenceGathererAgent()
            health_claim_state["category"] = "health"
            result = agent.process(health_claim_state)
            
            assert "search_results" in result
            assert "fact_check_results" in result
            assert len(result["fact_check_results"]) > 0
            assert "agent_messages" in result
            assert "agent_decisions" in result
    
    def test_process_empty_query(self, mock_config):
        """Test processing with empty query raises error"""
        from src.core.state import create_initial_state
        
        state = create_initial_state(
            query="",
            source_url="https://example.com"
        )
        
        with patch('src.agents.evidence_gatherer.get_config', return_value=mock_config), \
             patch('src.agents.evidence_gatherer.get_google_fact_check_client', side_effect=Exception("Not available")), \
             patch('src.agents.evidence_gatherer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = EvidenceGathererAgent()
            with pytest.raises(AgentProcessingError) as exc_info:
                agent.process(state)
            
            assert "empty" in str(exc_info.value).lower()
    
    def test_process_creates_message(self, health_claim_state, mock_config, mock_google_fact_check_client):
        """Test that process creates a message for LogicalAnalyzer"""
        with patch('src.agents.evidence_gatherer.get_config', return_value=mock_config), \
             patch('src.agents.evidence_gatherer.get_google_fact_check_client', return_value=mock_google_fact_check_client), \
             patch('src.agents.evidence_gatherer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = EvidenceGathererAgent()
            health_claim_state["category"] = "health"
            result = agent.process(health_claim_state)
            
            assert len(result["agent_messages"]) > 0
            message = result["agent_messages"][0]
            assert message["to_agent"] == "LogicalAnalyzer"
            assert message["message_type"] == "evidence"
            assert "search_results" in message["content"]
            assert "fact_check_results" in message["content"]
    
    def test_process_logs_decision(self, health_claim_state, mock_config, mock_google_fact_check_client):
        """Test that process logs a decision"""
        with patch('src.agents.evidence_gatherer.get_config', return_value=mock_config), \
             patch('src.agents.evidence_gatherer.get_google_fact_check_client', return_value=mock_google_fact_check_client), \
             patch('src.agents.evidence_gatherer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = EvidenceGathererAgent()
            health_claim_state["category"] = "health"
            result = agent.process(health_claim_state)
            
            assert agent.agent_name in result["agent_decisions"]
            decision = result["agent_decisions"][agent.agent_name]
            assert "decision" in decision
            assert "reasoning" in decision
    
    def test_gather_evidence_by_category_finance(self, mock_config):
        """Test evidence gathering for finance category"""
        with patch('src.agents.evidence_gatherer.get_config', return_value=mock_config), \
             patch('src.agents.evidence_gatherer.get_google_fact_check_client', side_effect=Exception("Not available")), \
             patch('src.agents.evidence_gatherer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = EvidenceGathererAgent()
            search_results, fact_check_results = agent._gather_evidence_by_category("finance", "Stock market claim")
            
            assert len(search_results) > 0
            assert len(fact_check_results) > 0
            assert any("SEC" in str(result) or "financial" in str(result).lower() for result in fact_check_results)
    
    def test_gather_evidence_by_category_politics(self, mock_config):
        """Test evidence gathering for politics category"""
        with patch('src.agents.evidence_gatherer.get_config', return_value=mock_config), \
             patch('src.agents.evidence_gatherer.get_google_fact_check_client', side_effect=Exception("Not available")), \
             patch('src.agents.evidence_gatherer.get_llama3_client', side_effect=Exception("Not available")):
            
            agent = EvidenceGathererAgent()
            search_results, fact_check_results = agent._gather_evidence_by_category("politics", "Election claim")
            
            assert len(search_results) > 0
            assert len(fact_check_results) > 0
