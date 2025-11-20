"""
Configuration management for the multi-agent system
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    enabled: bool = True
    timeout: Optional[float] = None
    retry_count: int = 3
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    console: bool = True


@dataclass
class Config:
    """Main configuration class"""
    # Agent configurations
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    
    # API configuration (Groq)
    llama3_api_key: Optional[str] = None
    llama3_api_url: Optional[str] = None
    llama3_model: str = "llama-3.3-70b-versatile"
    
    # Logging configuration
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Credibility thresholds
    low_credibility_threshold: int = 20
    high_credibility_threshold: int = 80
    
    # Verdict thresholds
    high_confidence_threshold: float = 0.95
    medium_confidence_threshold: float = 0.80
    low_confidence_threshold: float = 0.65
    
    # Domain lists
    low_credibility_domains: List[str] = field(default_factory=lambda: [
        "sketchy-site.net",
        "fake-news.com",
        "conspiracy-theory.org"
    ])
    
    high_credibility_domains: List[str] = field(default_factory=lambda: [
        "reuters.com",
        "bbc.com",
        "ap.org",
        "factcheck.org",
        "snopes.com"
    ])
    
    # Category keywords
    category_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "health": ["cancer", "cure", "disease", "medicine", "health", "medical"],
        "finance": ["stock", "market", "currency", "investment", "bank", "crypto"],
        "politics": ["election", "government", "policy", "politician", "vote"],
        "science": ["research", "study", "discovery", "scientific", "experiment"],
    })
    
    # Manipulative phrases
    manipulative_phrases: List[str] = field(default_factory=lambda: [
        "shocking", "you won't believe", "they don't want you to know",
        "secret", "hidden truth", "conspiracy", "cover-up"
    ])
    
    # Logical fallacy patterns
    fallacy_patterns: Dict[str, str] = field(default_factory=lambda: {
        "false_dilemma": r"(either|or|must|only)",
        "appeal_to_emotion": r"(feel|emotion|heart|fear)",
        "ad_hominem": r"(stupid|idiot|liar|corrupt)",
        "slippery_slope": r"(will lead to|inevitable|surely|certainly)"
    })
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary"""
        agents = {
            name: AgentConfig(**config) if isinstance(config, dict) else config
            for name, config in data.get("agents", {}).items()
        }
        
        logging_config = LoggingConfig(**data.get("logging", {}))
        
        return cls(
            agents=agents,
            logging=logging_config,
            llama3_api_key=data.get("llama3_api_key") or os.getenv("LLAMA3_API_KEY"),
            llama3_api_url=data.get("llama3_api_url") or os.getenv("LLAMA3_API_URL", "https://api.groq.com/openai/v1/chat/completions"),
            llama3_model=data.get("llama3_model", os.getenv("LLAMA3_MODEL", "llama-3.3-70b-versatile")),
            low_credibility_threshold=data.get("low_credibility_threshold", 20),
            high_credibility_threshold=data.get("high_credibility_threshold", 80),
            high_confidence_threshold=data.get("high_confidence_threshold", 0.95),
            medium_confidence_threshold=data.get("medium_confidence_threshold", 0.80),
            low_confidence_threshold=data.get("low_confidence_threshold", 0.65),
            low_credibility_domains=data.get("low_credibility_domains", []),
            high_credibility_domains=data.get("high_credibility_domains", []),
            category_keywords=data.get("category_keywords", {}),
            manipulative_phrases=data.get("manipulative_phrases", []),
            fallacy_patterns=data.get("fallacy_patterns", {})
        )
    
    @classmethod
    def from_file(cls, file_path: str) -> "Config":
        """Load config from JSON file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(path, "r") as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "agents": {
                name: {
                    "enabled": config.enabled,
                    "timeout": config.timeout,
                    "retry_count": config.retry_count,
                    "params": config.params
                }
                for name, config in self.agents.items()
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file": self.logging.file,
                "console": self.logging.console
            },
            "low_credibility_threshold": self.low_credibility_threshold,
            "high_credibility_threshold": self.high_credibility_threshold,
            "high_confidence_threshold": self.high_confidence_threshold,
            "medium_confidence_threshold": self.medium_confidence_threshold,
            "low_confidence_threshold": self.low_confidence_threshold,
            "low_credibility_domains": self.low_credibility_domains,
            "high_credibility_domains": self.high_credibility_domains,
            "category_keywords": self.category_keywords,
            "manipulative_phrases": self.manipulative_phrases,
            "fallacy_patterns": self.fallacy_patterns
        }


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = Config(
            llama3_api_key=os.getenv("LLAMA3_API_KEY"),
            llama3_api_url=os.getenv("LLAMA3_API_URL", "https://api.groq.com/openai/v1/chat/completions"),
            llama3_model=os.getenv("LLAMA3_MODEL", "llama-3.3-70b-versatile")
        )
    return _config


def load_config(file_path: Optional[str] = None) -> Config:
    """Load configuration from file or environment"""
    global _config
    
    if file_path:
        _config = Config.from_file(file_path)
    else:
        # Try to load from environment variable
        config_path = os.getenv("AGENT_CONFIG_PATH")
        if config_path and Path(config_path).exists():
            _config = Config.from_file(config_path)
        else:
            _config = Config()
    
    return _config

