"""
Streamlit Web UI for Multi-Agent Disinformation Detector
Run with: streamlit run app.py
"""
import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import run_detector

# Page configuration
st.set_page_config(
    page_title="Disinformation Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .verdict-true {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .verdict-false {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .verdict-misleading {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .verdict-unverifiable {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

def get_verdict_style(verdict: str) -> str:
    """Get CSS class for verdict styling"""
    verdict_lower = verdict.lower()
    if "false" in verdict_lower:
        return "verdict-false"
    elif "true" in verdict_lower:
        return "verdict-true"
    elif "misleading" in verdict_lower:
        return "verdict-misleading"
    elif "unverifiable" in verdict_lower:
        return "verdict-unverifiable"
    return ""

def format_confidence_color(confidence: float) -> str:
    """Get color based on confidence score"""
    if confidence >= 0.8:
        return "green"
    elif confidence >= 0.6:
        return "orange"
    else:
        return "red"

def main():
    # Header
    st.markdown('<div class="main-header">🔍 Multi-Agent Disinformation Detector</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")
        
        # Example claims
        st.subheader("📋 Example Claims")
        
        example_claims = {
            "Low Credibility": {
                "claim": "New cure for cancer discovered today.",
                "source": "http://www.sketchy-site.net"
            },
            "High Credibility": {
                "claim": "New study shows potential benefits of exercise.",
                "source": "https://www.reuters.com"
            },
            "Health Claim": {
                "claim": "Vaccines cause autism in children.",
                "source": "http://conspiracy-theory.org"
            },
            "Science Claim": {
                "claim": "Climate change is a natural phenomenon, not caused by humans.",
                "source": "https://www.bbc.com"
            }
        }
        
        selected_example = st.selectbox("Load Example:", ["None"] + list(example_claims.keys()))
        
        st.markdown("---")
        st.markdown("""
        ### ℹ️ How to Use
        1. Enter a news claim or statement
        2. Provide the source URL
        3. Click "Analyze Claim" to start detection
        4. View detailed results and analysis
        
        ### 🔍 What It Does
        - Classifies claim category
        - Evaluates source credibility
        - Gathers evidence using Llama3.3 70B
        - Analyzes sentiment and logical fallacies
        - Synthesizes final verdict with confidence score
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Claim to Analyze")
        
        # Form inputs
        claim = st.text_area(
            "Claim/Statement:",
            height=100,
            placeholder="Enter the news claim or statement you want to verify..."
        )
        
        source_url = st.text_input(
            "Source URL:",
            placeholder="https://example.com/article"
        )
    
    with col2:
        st.subheader("🎯 Quick Stats")
        if st.session_state.get('last_result'):
            result = st.session_state.last_result
            st.metric("Credibility", f"{result.get('credibility_score', 0)}/100")
            st.metric("Confidence", f"{result.get('confidence_score', 0.0):.1%}")
            st.metric("Category", result.get('category', 'N/A'))
            st.metric("Agents", len(result.get('agent_decisions', {})))
    
    # Load example if selected
    if selected_example != "None" and selected_example in example_claims:
        example = example_claims[selected_example]
        claim = example["claim"]
        source_url = example["source"]
        # Update text inputs (requires rerun)
        st.rerun()
    
    # Analyze button
    analyze_button = st.button("🔍 Analyze Claim", type="primary", use_container_width=True)
    
    # Analysis results
    if analyze_button:
        if not claim.strip():
            st.error("❌ Please enter a claim to analyze.")
        elif not source_url.strip():
            st.error("❌ Please enter a source URL.")
        else:
            # Show progress
            with st.spinner("🤖 Multi-agent analysis in progress... This may take a moment."):
                try:
                    # Run detector (with minimal logging in UI)
                    result = run_detector(claim, source_url, verbose=False)
                    st.session_state.last_result = result
                    
                    # Success message
                    st.success("✅ Analysis complete!")
                    
                    # Display results
                    display_results(result)
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)
    
    # Show last result if available
    elif st.session_state.get('last_result'):
        st.info("📊 Showing last analysis result. Click 'Analyze Claim' to analyze a new claim.")
        display_results(st.session_state.last_result)

def display_results(result: Dict[str, Any]):
    """Display analysis results in a formatted way"""
    
    verdict = result.get('final_verdict', 'N/A')
    confidence = result.get('confidence_score', 0.0)
    credibility = result.get('credibility_score', 0)
    category = result.get('category', 'N/A')
    reasoning = result.get('reasoning', 'N/A')
    evidence = result.get('evidence_summary', 'N/A')
    requires_review = result.get('requires_review', False)
    review_reason = result.get('review_reason', '')
    
    # Verdict display
    verdict_style = get_verdict_style(verdict)
    st.markdown(f'<div class="result-box {verdict_style}">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### 🎯 Final Verdict: {verdict}")
    
    with col2:
        confidence_color = format_confidence_color(confidence)
        st.markdown(f'<p style="font-size: 1.2rem;"><strong>Confidence:</strong> <span style="color: {confidence_color};">{confidence:.1%}</span></p>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<p style="font-size: 1.2rem;"><strong>Credibility:</strong> {credibility}/100</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Metrics row
    st.markdown("### 📊 Analysis Metrics")
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Category", category)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metrics_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Credibility Score", f"{credibility}/100")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metrics_col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Confidence", f"{confidence:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metrics_col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        agent_count = len(result.get('agent_decisions', {}))
        st.metric("Agents Active", agent_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tabs for detailed information
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Reasoning", "🔍 Evidence", "🤖 Agent Details", "📋 Raw Data"])
    
    with tab1:
        st.markdown("### Reasoning")
        st.info(reasoning)
        
        if requires_review:
            st.warning(f"⚠️ **Review Required:** {review_reason}")
    
    with tab2:
        st.markdown("### Evidence Summary")
        st.text_area("", evidence, height=200, disabled=True)
        
        # Fact check results
        fact_checks = result.get('fact_check_results', [])
        if fact_checks:
            st.markdown("### Fact-Check Results")
            for i, fc in enumerate(fact_checks, 1):
                with st.expander(f"Source {i}: {fc.get('source', 'Unknown')}"):
                    st.write(f"**Verdict:** {fc.get('verdict', 'N/A')}")
                    st.write(f"**Confidence:** {fc.get('confidence', 0.0):.1%}")
                    if fc.get('reasoning'):
                        st.write(f"**Reasoning:** {fc.get('reasoning')}")
    
    with tab3:
        st.markdown("### Agent Decisions")
        agent_decisions = result.get('agent_decisions', {})
        
        for agent_name, decision in agent_decisions.items():
            with st.expander(f"🤖 {agent_name}"):
                if isinstance(decision, dict):
                    st.write(f"**Decision:** {decision.get('decision', 'N/A')}")
                    st.write(f"**Reasoning:** {decision.get('reasoning', 'N/A')}")
                    st.write(f"**Confidence:** {decision.get('confidence', 0.0):.1%}")
                else:
                    st.write(decision)
        
        # Messages
        messages = result.get('agent_messages', [])
        if messages:
            st.markdown("### Agent Messages")
            st.write(f"Total messages exchanged: {len(messages)}")
            for i, msg in enumerate(messages[:5], 1):  # Show first 5
                with st.expander(f"Message {i}: {msg.get('from_agent', 'Unknown')} → {msg.get('to_agent', 'Unknown')}"):
                    st.write(f"**Type:** {msg.get('message_type', 'N/A')}")
                    st.json(msg.get('content', {}))
    
    with tab4:
        st.markdown("### Raw State Data")
        st.json(result)
    
    # Additional information
    with st.expander("📈 Additional Analysis Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Sentiment Analysis")
            sentiment = result.get('sentiment_analysis', {})
            if sentiment:
                st.write(f"**Sentiment:** {sentiment.get('sentiment', 'N/A')}")
                st.write(f"**Manipulation Score:** {sentiment.get('manipulation_score', 0.0):.2f}")
                st.write(f"**Emotional Appeal:** {'Yes' if sentiment.get('is_emotional_appeal') else 'No'}")
        
        with col2:
            st.markdown("#### Logical Analysis")
            fallacies = result.get('logical_fallacies', [])
            st.write(f"**Fallacies Detected:** {len(fallacies)}")
            if fallacies:
                for fallacy in fallacies:
                    st.write(f"- {fallacy}")
            
            contradiction = result.get('internal_contradiction', False)
            st.write(f"**Internal Contradiction:** {'Yes' if contradiction else 'No'}")

if __name__ == "__main__":
    main()

