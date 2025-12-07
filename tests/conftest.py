"""
Shared pytest fixtures for agent tests
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.state import AgentState, create_initial_state
from src.config.config import Config, LoggingConfig


@pytest.fixture
def sample_state() -> AgentState:
    """Create a sample agent state for testing"""
    return create_initial_state(
        query="New cure for cancer discovered today.",
        source_url="http://www.sketchy-site.net"
    )


@pytest.fixture
def health_claim_state() -> AgentState:
    """Create a state with a health-related claim"""
    return create_initial_state(
        query="Vaccines cause autism in children.",
        source_url="http://conspiracy-theory.org"
    )


@pytest.fixture
def finance_claim_state() -> AgentState:
    """Create a state with a finance-related claim"""
    return create_initial_state(
        query="Stock market will crash tomorrow.",
        source_url="https://www.reuters.com"
    )


@pytest.fixture
def high_credibility_state() -> AgentState:
    """Create a state with a high credibility source"""
    return create_initial_state(
        query="New study shows potential benefits of exercise.",
        source_url="https://www.bbc.com"
    )


@pytest.fixture
def low_credibility_state() -> AgentState:
    """Create a state with a low credibility source"""
    return create_initial_state(
        query="Shocking secret they don't want you to know!",
        source_url="http://www.sketchy-site.net"
    )


@pytest.fixture
def mock_config() -> Config:
    """Create a mock configuration"""
    return Config(
        low_credibility_threshold=20,
        high_credibility_threshold=80,
        high_confidence_threshold=0.95,
        medium_confidence_threshold=0.80,
        low_confidence_threshold=0.65,
        low_credibility_domains=["sketchy-site.net", "fake-news.com"],
        high_credibility_domains=["reuters.com", "bbc.com", "ap.org"],
        category_keywords={
            "health": ["cancer", "cure", "disease", "medicine", "health", "medical"],
            "finance": ["stock", "market", "currency", "investment", "bank", "crypto"],
            "politics": ["election", "government", "policy", "politician", "vote"],
            "science": ["research", "study", "discovery", "scientific", "experiment"],
        },
        manipulative_phrases=[
            "shocking", "you won't believe", "they don't want you to know",
            "secret", "hidden truth", "conspiracy", "cover-up"
        ],
        fallacy_patterns={
            "false_dilemma": r"(either|or|must|only)",
            "appeal_to_emotion": r"(feel|emotion|heart|fear)",
            "ad_hominem": r"(stupid|idiot|liar|corrupt)",
            "slippery_slope": r"(will lead to|inevitable|surely|certainly)"
        }
    )


@pytest.fixture
def mock_llama3_client():
    """Create a mock Llama3 client"""
    mock_client = Mock()
    
    # Default responses
    mock_client.generate.return_value = '{"verdict": "FALSE", "confidence": 0.85, "reasoning": "Test reasoning"}'
    mock_client.analyze_claim.return_value = {
        "verdict": "FALSE",
        "confidence": 0.85,
        "reasoning": "Test reasoning",
        "evidence": "Test evidence"
    }
    
    return mock_client


@pytest.fixture
def mock_google_fact_check_client():
    """Create a mock Google Fact Check client"""
    mock_client = Mock()
    
    # Default responses
    mock_client.search_claims.return_value = [
        {
            "claim_text": "Test claim",
            "source": "Snopes",
            "verdict": "FALSE",
            "confidence": 0.90,
            "url": "https://example.com",
            "review_date": "2024-01-01"
        }
    ]
    
    mock_client.check_claim.return_value = {
        "found": True,
        "verdict": "FALSE",
        "confidence": 0.85,
        "reasoning": "Found fact-check results",
        "results": [
            {
                "source": "Snopes",
                "verdict": "FALSE",
                "confidence": 0.90,
                "url": "https://example.com"
            }
        ],
        "source_count": 1
    }
    
    return mock_client


@pytest.fixture
def mock_llama3_sentiment_response():
    """Mock Llama3 sentiment analysis response"""
    return {
        "sentiment": "negative",
        "is_emotional_appeal": True,
        "manipulative_phrases": ["shocking", "secret"],
        "manipulation_score": 0.75
    }


@pytest.fixture
def mock_llama3_logical_response():
    """Mock Llama3 logical analysis response"""
    return {
        "fallacies": ["false_dilemma", "appeal_to_emotion"],
        "has_contradictions": True,
        "reasoning_quality": "poor"
    }


@pytest.fixture
def mock_fact_check_results():
    """Mock fact-check results"""
    return [
        {
            "source": "Snopes",
            "verdict": "FALSE",
            "confidence": 0.90,
            "reasoning": "Fact-checked as false",
            "url": "https://snopes.com/fact-check/example"
        },
        {
            "source": "FactCheck.org",
            "verdict": "FALSE",
            "confidence": 0.85,
            "reasoning": "Verified as false",
            "url": "https://factcheck.org/example"
        }
    ]


@pytest.fixture
def state_with_evidence() -> AgentState:
    """Create a state with evidence already gathered"""
    state = create_initial_state(
        query="New cure for cancer discovered today.",
        source_url="http://www.sketchy-site.net"
    )
    state["search_results"] = [
        "No peer-reviewed study supports this claim",
        "Major medical institutions deny this claim"
    ]
    state["fact_check_results"] = [
        {"source": "WHO", "verdict": "False", "confidence": 0.95},
        {"source": "Snopes", "verdict": "False", "confidence": 0.90}
    ]
    state["category"] = "health"
    state["credibility_score"] = 15
    return state


@pytest.fixture
def state_with_sentiment() -> AgentState:
    """Create a state with sentiment analysis"""
    state = create_initial_state(
        query="Shocking secret they don't want you to know!",
        source_url="http://www.sketchy-site.net"
    )
    state["sentiment_analysis"] = {
        "sentiment": "negative",
        "manipulation_score": 0.75,
        "is_emotional_appeal": True,
        "manipulative_phrases_found": 2
    }
    return state


@pytest.fixture
def state_with_fallacies() -> AgentState:
    """Create a state with logical fallacies"""
    state = create_initial_state(
        query="Either you believe this or you're stupid.",
        source_url="http://www.sketchy-site.net"
    )
    state["logical_fallacies"] = ["false_dilemma", "ad_hominem"]
    state["internal_contradiction"] = True
    return state


@pytest.fixture
def state_complete() -> AgentState:
    """Create a complete state with all fields populated"""
    state = create_initial_state(
        query="New cure for cancer discovered today.",
        source_url="http://www.sketchy-site.net"
    )
    state["category"] = "health"
    state["credibility_score"] = 15
    state["search_results"] = ["Evidence 1", "Evidence 2"]
    state["fact_check_results"] = [
        {"source": "WHO", "verdict": "False", "confidence": 0.95}
    ]
    state["sentiment_analysis"] = {
        "sentiment": "positive",
        "manipulation_score": 0.3
    }
    state["cross_references"] = [
        {"source": "WHO", "verdict": "False", "confidence": 0.95}
    ]
    state["logical_fallacies"] = []
    state["internal_contradiction"] = True
    return state
