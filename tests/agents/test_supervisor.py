"""
Tests for SupervisorAgent
"""
import pytest
from unittest.mock import patch
from src.agents.supervisor import SupervisorAgent


class TestSupervisorAgent:
    """Test suite for SupervisorAgent"""
    
    def test_initialization(self, mock_config):
        """Test agent initialization"""
        with patch('src.agents.supervisor.get_config', return_value=mock_config):
            agent = SupervisorAgent()
            assert agent.agent_name == "Supervisor"
            assert agent.agent_type == "coordination"
    
    def test_analyze_coordination(self, state_complete, mock_config):
        """Test coordination analysis"""
        with patch('src.agents.supervisor.get_config', return_value=mock_config):
            agent = SupervisorAgent()
            result = agent._analyze_coordination(state_complete)
            
            assert "total_agents_active" in result
            assert "messages_received" in result
            assert "coordination_status" in result
            assert result["coordination_status"] == "successful"
    
    def test_analyze_coordination_with_review(self, state_complete, mock_config):
        """Test coordination analysis when review is required"""
        with patch('src.agents.supervisor.get_config', return_value=mock_config):
            agent = SupervisorAgent()
            state_complete["requires_review"] = True
            state_complete["review_reason"] = "Low confidence"
            
            result = agent._analyze_coordination(state_complete)
            
            assert result["action_required"] == "review_needed"
            assert result["reason"] == "Low confidence"
    
    def test_analyze_coordination_no_review(self, state_complete, mock_config):
        """Test coordination analysis when no review is needed"""
        with patch('src.agents.supervisor.get_config', return_value=mock_config):
            agent = SupervisorAgent()
            state_complete["requires_review"] = False
            
            result = agent._analyze_coordination(state_complete)
            
            assert result["action_required"] == "none"
    
    def test_process_coordinates_agents(self, state_complete, mock_config):
        """Test that process coordinates agent activities"""
        with patch('src.agents.supervisor.get_config', return_value=mock_config):
            agent = SupervisorAgent()
            result = agent.process(state_complete)
            
            assert "agent_decisions" in result
            assert "workflow_stage" in result
    
    def test_process_preserves_state(self, state_complete, mock_config):
        """Test that process preserves important state fields"""
        with patch('src.agents.supervisor.get_config', return_value=mock_config):
            agent = SupervisorAgent()
            original_decisions = state_complete.get("agent_decisions", {})
            original_stage = state_complete.get("workflow_stage", "")
            
            result = agent.process(state_complete)
            
            assert result["agent_decisions"] == original_decisions
            assert result["workflow_stage"] == original_stage
    
    def test_coordination_counts_agents(self, state_complete, mock_config):
        """Test that coordination correctly counts active agents"""
        with patch('src.agents.supervisor.get_config', return_value=mock_config):
            agent = SupervisorAgent()
            state_complete["agent_decisions"] = {
                "TriageManager": {},
                "SourceScorer": {},
                "EvidenceGatherer": {}
            }
            
            result = agent._analyze_coordination(state_complete)
            assert result["total_agents_active"] == 3
    
    def test_coordination_counts_messages(self, state_complete, mock_config):
        """Test that coordination correctly counts messages"""
        with patch('src.agents.supervisor.get_config', return_value=mock_config):
            agent = SupervisorAgent()
            state_complete["agent_messages"] = [
                {"to_agent": "Supervisor", "from_agent": "TriageManager"},
                {"to_agent": "Supervisor", "from_agent": "SourceScorer"},
                {"to_agent": "OtherAgent", "from_agent": "TriageManager"}
            ]
            
            result = agent._analyze_coordination(state_complete)
            assert result["messages_received"] == 2  # Only messages to Supervisor
