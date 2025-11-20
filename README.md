# Multi-Agent Disinformation Detector

A production-ready, scalable multi-agent system for detecting disinformation and fake news using LangGraph orchestration. The system employs specialized agents that work together to analyze claims, assess source credibility, gather evidence, and synthesize final verdicts.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Verdict Types](#verdict-types)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🎯 Overview

### What is this?

The Multi-Agent Disinformation Detector is an intelligent system that uses multiple specialized agents to analyze news claims and determine their veracity. Each agent has a specific role, from classifying claims to assessing source credibility, gathering evidence, and synthesizing final verdicts.

### Key Capabilities

- **Claim Classification**: Automatically categorizes claims into health, finance, politics, science, etc.
- **Source Credibility Assessment**: Evaluates source credibility based on domain reputation
- **Evidence Gathering**: Gathers evidence and fact-check results from multiple sources
- **Sentiment Analysis**: Analyzes sentiment and detects emotional manipulation
- **Cross-Referencing**: Cross-references with multiple sources for consensus
- **Logical Analysis**: Detects logical fallacies and contradictions
- **Verdict Synthesis**: Synthesizes final verdicts with confidence scores

## ✨ Features

- **Multi-Agent Architecture**: 8 specialized agents working together
- **Workflow Orchestration**: LangGraph-based workflow with conditional routing
- **Production-Ready**: Built with error handling, logging, and validation
- **Scalable**: Modular architecture that can scale horizontally
- **Configurable**: Easy configuration through JSON files or environment variables
- **Type-Safe**: Full type hints and validation

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip
- Virtual environment (recommended)

### Installation Steps

1. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import langgraph; print('Installation successful!')"
```

### Troubleshooting Dependency Conflicts

If you encounter dependency conflicts, create a fresh virtual environment:

```bash
# Create new virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: This project uses `langgraph>=1.0.0` which requires `langchain-core>=1.0.0` and `pydantic>=2.0.0`. If you have older langchain packages installed, they may conflict.

## ⚡ Quick Start

### Run the Web UI (Recommended)

The easiest way to test the system is using the web interface:

```bash
./run_ui.sh
```

Or directly:
```bash
python -m streamlit run app.py
```

The UI will open in your browser at `http://localhost:8501`

### Run the Application (CLI)

```bash
python main.py
```

### Use Programmatically

```python
from main import run_detector

claim = "New cure for cancer discovered today."
source = "http://www.sketchy-site.net"

result = run_detector(claim, source)
print(result['final_verdict'])
print(f"Confidence: {result['confidence_score']:.2%}")
```

## 📖 Usage

### Basic Usage

```python
from main import run_detector

claim = "New cure for cancer discovered today."
source = "http://www.sketchy-site.net"

result = run_detector(claim, source)
print(result['final_verdict'])
```

### With Configuration

```python
from main import run_detector

claim = "New cure for cancer discovered today."
source = "http://www.sketchy-site.net"

result = run_detector(
    claim,
    source,
    config_path="config.json",
    verbose=True
)
```

### Custom Workflow

```python
from src.workflow import create_workflow
from src.core.state import create_initial_state

# Create workflow
workflow = create_workflow()

# Run workflow
initial_state = create_initial_state(
    query="New cure for cancer discovered today.",
    source_url="http://www.sketchy-site.net"
)

final_state = workflow.run(
    initial_state["query"],
    initial_state["source_url"]
)

print(f"Verdict: {final_state['final_verdict']}")
print(f"Confidence: {final_state['confidence_score']:.2%}")
```

## ⚙️ Configuration

### Configuration File

Create a `config.json` file:

```json
{
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/app.log",
    "console": true
  },
  "low_credibility_threshold": 20,
  "high_credibility_threshold": 80,
  "high_confidence_threshold": 0.95,
  "medium_confidence_threshold": 0.80,
  "low_confidence_threshold": 0.65
}
```

### Environment Variables

You can also set configuration via environment variables:

```bash
export AGENT_CONFIG_PATH="config.json"
export LOG_LEVEL="INFO"
```

### Configuration Options

- **logging**: Logging configuration (level, format, file, console)
- **low_credibility_threshold**: Threshold for low credibility sources (default: 20)
- **high_credibility_threshold**: Threshold for high credibility sources (default: 80)
- **high_confidence_threshold**: Threshold for high confidence verdicts (default: 0.95)
- **medium_confidence_threshold**: Threshold for medium confidence verdicts (default: 0.80)
- **low_confidence_threshold**: Threshold for low confidence verdicts (default: 0.65)
- **low_credibility_domains**: List of low credibility domains
- **high_credibility_domains**: List of high credibility domains
- **category_keywords**: Keywords for category classification
- **manipulative_phrases**: Phrases that indicate emotional manipulation
- **fallacy_patterns**: Patterns for detecting logical fallacies

## 📊 Verdict Types

The system can produce the following verdict types:

### FALSE (High Confidence)
- **Conditions**: Low credibility source (< 20) AND internal contradictions detected
- **Confidence**: 0.95
- **Example**: Known disinformation source with contradictions

### FALSE (Medium Confidence)
- **Conditions**: Fact-checkers rate as false
- **Confidence**: 0.80
- **Example**: Multiple fact-checkers contradict the claim

### MISLEADING
- **Conditions**: Contradictory evidence or logical fallacies detected
- **Confidence**: 0.65
- **Example**: Evidence contradicts the claim or logical fallacies are present

### UNVERIFIABLE
- **Conditions**: Insufficient evidence available
- **Confidence**: 0.40
- **Example**: No evidence available to make a determination

## 🏗️ How It Works

### Agents

The system uses 8 specialized agents:

1. **TriageManagerAgent** - Classifies claims into categories (health, finance, politics, science, etc.)
2. **SourceScorerAgent** - Evaluates source credibility based on domain reputation
3. **EvidenceGathererAgent** - Gathers evidence and fact-check results
4. **SentimentAnalyzerAgent** - Analyzes sentiment and detects emotional manipulation
5. **CrossReferenceAgent** - Cross-references with multiple sources for consensus
6. **LogicalAnalyzerAgent** - Detects logical fallacies and contradictions
7. **VerdictSynthesizerAgent** - Synthesizes final verdict with confidence scores
8. **SupervisorAgent** - Coordinates agent activities and handles review requests

### Workflow

The system follows this workflow:

1. **Triage**: Classifies the claim into a category
2. **Parallel Processing**: Evaluates source credibility, gathers evidence, and analyzes sentiment simultaneously
3. **Cross-Referencing**: Cross-references evidence with multiple sources
4. **Analysis**: Analyzes logical consistency and detects fallacies
5. **Routing**: Routes to fast verdict or full analysis based on credibility
6. **Synthesis**: Synthesizes final verdict with confidence score
7. **Review**: Flags cases requiring human review if needed

## 🌐 Web UI

### Features

The Streamlit web interface provides an easy way to test the system:

- **📝 Input Form**: Enter claims and source URLs to analyze
- **📊 Real-time Results**: View detailed analysis results
- **🎯 Verdict Display**: Color-coded verdicts (TRUE/FALSE/MISLEADING/UNVERIFIABLE)
- **📈 Metrics Dashboard**: See credibility scores, confidence levels, and more
- **🔍 Detailed Analysis**: Explore reasoning, evidence, agent decisions, and raw data
- **📋 Example Claims**: Quick access to pre-loaded example claims

### How to Use

1. **Start the UI**: Run `./run_ui.sh` or `python -m streamlit run app.py`
2. **Enter a Claim**: Type or paste the news claim you want to verify
3. **Provide Source URL**: Enter the website URL where the claim originated
4. **Click "Analyze Claim"**: The multi-agent system will process the claim
5. **View Results**: See the verdict, confidence score, reasoning, and detailed analysis

## 🔌 APIs

### Current Status

The system is integrated with **Groq's Llama3.3 70B API** for intelligent analysis. Configure your API key in the `.env` file:

```env
LLAMA3_API_KEY=your_groq_api_key_here
LLAMA3_API_URL=https://api.groq.com/openai/v1/chat/completions
LLAMA3_MODEL=llama-3.3-70b-versatile
```

### Additional APIs (Planned)

- **Fact-Checking APIs**: Google Fact Check API, PolitiFact API, Snopes API
- **Search APIs**: Google Search API, Bing Search API
- **Source Credibility APIs**: Media Bias/Fact Check API, NewsGuard API
- **News APIs**: NewsAPI

### Optional APIs (Planned)

- **LLM APIs**: OpenAI API, Anthropic API (for advanced analysis)
- **Sentiment Analysis APIs**: Google Cloud Natural Language API
- **Database APIs**: PostgreSQL (for data persistence)
- **Cache APIs**: Redis (for caching)

## 🐛 Troubleshooting

### ModuleNotFoundError: No module named 'langgraph'

**Problem**: Error when running Streamlit UI

**Solution**: Use the correct Python environment:
```bash
# Option 1: Use the run script (recommended)
./run_ui.sh

# Option 2: Use Python module syntax
python -m streamlit run app.py
```

**Note**: Don't use `streamlit run app.py` directly - it may use a different Python environment.

### Dependency Conflicts

**Problem**: Dependency conflicts with langchain packages

**Solution**: Create a fresh virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Import Errors

**Problem**: `ModuleNotFoundError`

**Solution**: 
1. Verify installation: `pip install -r requirements.txt`
2. Check Python version: `python --version` (should be 3.10+)
3. Check virtual environment activation

### Configuration Errors

**Problem**: Configuration file not found or invalid

**Solution**:
1. Create `config.json` file
2. Verify JSON syntax
3. Check file path

### API Key Not Found

**Problem**: Llama3 API key not configured

**Solution**: Create a `.env` file with your Groq API key:
```env
LLAMA3_API_KEY=your_groq_api_key_here
LLAMA3_API_URL=https://api.groq.com/openai/v1/chat/completions
LLAMA3_MODEL=llama-3.3-70b-versatile
```

### Port Already in Use

**Problem**: Port 8501 is already in use

**Solution**: Use a different port:
```bash
streamlit run app.py --server.port 8502
```


For questions, issues, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/fraud-news-detector-langgraph-multiagents/issues)
- **Discussions**: [Join discussions](https://github.com/yourusername/fraud-news-detector-langgraph-multiagents/discussions)
- **Email**: your.email@example.com

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **LangGraph** - Workflow orchestration
- **LangChain** - Agent framework
- **Pydantic** - Data validation

## 🔗 Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
