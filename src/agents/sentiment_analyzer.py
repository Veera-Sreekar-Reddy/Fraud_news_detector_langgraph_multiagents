"""
Sentiment Analyzer Agent
Responsible for analyzing sentiment and emotional manipulation
"""
from typing import Dict, Any, Optional
from ..core.base_agent import BaseAgent
from ..core.state import AgentState
from ..core.exceptions import AgentProcessingError
from ..config import get_config
from ..integrations import get_llama3_client


class SentimentAnalyzerAgent(BaseAgent):
    """Agent responsible for analyzing sentiment and emotional manipulation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Sentiment Analyzer Agent
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__("SentimentAnalyzer", "sentiment_analysis")
        self.config = get_config()
        self.manipulative_phrases = (
            config.get("manipulative_phrases", []) if config
            else self.config.manipulative_phrases
        )
        self.positive_words = ["good", "great", "amazing", "cure", "discovered", "breakthrough"]
        self.negative_words = ["danger", "warning", "threat", "scam", "fake"]
        
        try:
            self.llama3_client = get_llama3_client()
            self.use_llama3 = True
        except Exception as e:
            self.logger.warning(f"Llama3 API not available, using fallback: {e}")
            self.llama3_client = None
            self.use_llama3 = False
    
    def _analyze_sentiment(self, query: str) -> Dict[str, Any]:
        """
        Analyze sentiment and emotional manipulation using Llama3.3 70B
        
        Args:
            query: Claim query
            
        Returns:
            Sentiment analysis dictionary
        """
        # Use Llama3 API if available
        if self.use_llama3 and self.llama3_client:
            try:
                self.logger.info("Using Llama3.3 70B for sentiment analysis")
                
                analysis = self.llama3_client.analyze_claim(
                    claim=query,
                    analysis_type="sentiment"
                )
                
                if isinstance(analysis, dict) and "sentiment" in analysis:
                    # Map Llama3 response to our format
                    sentiment = analysis.get("sentiment", "neutral").lower()
                    if sentiment not in ["positive", "negative", "neutral"]:
                        sentiment = "neutral"
                    
                    manipulation_score = float(analysis.get("manipulation_score", 0.0))
                    manipulative_phrases_found = len(analysis.get("manipulative_phrases", []))
                    
                    return {
                        "sentiment": sentiment,
                        "manipulation_score": manipulation_score,
                        "manipulative_phrases_found": manipulative_phrases_found,
                        "is_emotional_appeal": analysis.get("is_emotional_appeal", manipulation_score > 0.5),
                        "positive_words": 0,  # Not returned by Llama3
                        "negative_words": 0   # Not returned by Llama3
                    }
                
            except Exception as e:
                self.logger.warning(f"Llama3 API call failed, using fallback: {e}")
                # Fall through to fallback
        
        # Fallback to rule-based analysis
        self.logger.info("Using fallback sentiment analysis")
        query_lower = query.lower()
        
        # Detect manipulative language
        manipulative_count = sum(
            1 for phrase in self.manipulative_phrases
            if phrase.lower() in query_lower
        )
        
        # Determine sentiment
        positive_count = sum(1 for word in self.positive_words if word in query_lower)
        negative_count = sum(1 for word in self.negative_words if word in query_lower)
        
        sentiment = "neutral"
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        
        # Calculate manipulation score
        manipulation_score = min(manipulative_count * 0.3, 1.0)
        
        return {
            "sentiment": sentiment,
            "manipulation_score": manipulation_score,
            "manipulative_phrases_found": manipulative_count,
            "is_emotional_appeal": manipulation_score > 0.5,
            "positive_words": positive_count,
            "negative_words": negative_count
        }
    
    def process(self, state: AgentState) -> Dict[str, Any]:
        """
        Process state and analyze sentiment
        
        Args:
            state: Current agent state
            
        Returns:
            Dictionary with state updates
            
        Raises:
            AgentProcessingError: If processing fails
        """
        try:
            self.logger.info("Analyzing sentiment and emotional manipulation")
            query = state.get("query", "")
            
            if not query:
                raise AgentProcessingError("Query is empty", agent_name=self.agent_name)
            
            # Analyze sentiment
            sentiment_analysis = self._analyze_sentiment(query)
            
            self.logger.info(
                f"Sentiment: {sentiment_analysis['sentiment']}, "
                f"Manipulation score: {sentiment_analysis['manipulation_score']:.2f}"
            )
            
            # Send message to LogicalAnalyzer
            message = self.send_message(
                to_agent="LogicalAnalyzer",
                message_type="sentiment",
                content=sentiment_analysis,
                confidence=0.75
            )
            
            decision = self.log_decision(
                decision=(
                    f"Sentiment: {sentiment_analysis['sentiment']}, "
                    f"Manipulation: {sentiment_analysis['manipulation_score']:.2f}"
                ),
                reasoning=f"Found {sentiment_analysis['manipulative_phrases_found']} manipulative phrases",
                confidence=0.75
            )
            
            return {
                "sentiment_analysis": sentiment_analysis,
                "agent_messages": [message],  # Reducer will merge
                "agent_decisions": {self.agent_name: decision}  # Reducer will merge
            }
        except Exception as e:
            error_msg = f"Failed to analyze sentiment: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise AgentProcessingError(error_msg, agent_name=self.agent_name) from e

