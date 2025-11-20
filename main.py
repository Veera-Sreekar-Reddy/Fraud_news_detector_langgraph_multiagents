"""
Multi-Agent Disinformation Detector
Production-level implementation with scalable architecture
"""
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.workflow import create_workflow
from src.core.state import create_initial_state
from src.utils.logger import setup_logging
from src.utils.validators import validate_query, validate_url
from src.config import load_config

logger = logging.getLogger(__name__)


def run_detector(
    claim: str,
    source_url: str,
    config_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run the disinformation detector on a claim
    
    Args:
        claim: Claim to check
        source_url: Source URL
        config_path: Optional path to configuration file
        verbose: Whether to print detailed output
        
    Returns:
        Final agent state
        
    Raises:
        ValueError: If claim or source_url is invalid
    """
    try:
        # Setup logging
        if config_path:
            config = load_config(config_path)
        else:
            config = load_config()
        
        setup_logging(
            level=config.logging.level,
            format_string=config.logging.format,
            log_file=config.logging.file,
            console=config.logging.console
        )
        
        # Validate inputs
        validate_query(claim)
        validate_url(source_url)
        
        # Create workflow
        workflow = create_workflow()
        
        # Run workflow
        if verbose:
            print("\n" + "="*60)
            print("--- STARTING MULTI-AGENT DISINFORMATION DETECTOR ---")
            print("="*60)
        
        final_state = workflow.run(claim, source_url)
        
        if verbose:
            print("\n" + "="*60)
            print("--- WORKFLOW EXECUTION COMPLETE ---")
            print("="*60)
            print(f"\n📝 Claim: {final_state['query']}")
            print(f"🔗 Source URL: {final_state['source_url']}")
            print(f"📂 Category: {final_state.get('category', 'N/A')}")
            print(f"⭐ Credibility Score: {final_state.get('credibility_score', 0)}/100")
            print(f"🎯 Final Verdict: {final_state.get('final_verdict', 'N/A')}")
            print(f"📊 Confidence Score: {final_state.get('confidence_score', 0.0):.2%}")
            print(f"🧠 Reasoning: {final_state.get('reasoning', 'N/A')}")
            print(f"🔍 Evidence Summary:\n{final_state.get('evidence_summary', 'N/A')}")
            
            if final_state.get('requires_review'):
                print(f"\n⚠️ Review Required: {final_state.get('review_reason', 'Unknown reason')}")
            
            print(f"\n🤖 Agents Active: {len(final_state.get('agent_decisions', {}))}")
            print(f"📨 Messages Exchanged: {len(final_state.get('agent_messages', []))}")
            print("="*60 + "\n")
        
        logger.info("Detector execution completed successfully")
        
        return final_state
    except Exception as e:
        logger.error(f"Detector execution failed: {str(e)}", exc_info=True)
        if verbose:
            print(f"\n❌ Error: {str(e)}\n")
        raise


def main():
    """Main entry point"""
    # Example 1: Low credibility source
    print("\n🔍 Example 1: Low Credibility Source")
    claim1 = "New cure for cancer discovered today."
    source1 = "http://www.sketchy-site.net"
    run_detector(claim1, source1)
    
    # Example 2: High credibility source (simulated)
    print("\n🔍 Example 2: High Credibility Source")
    claim2 = "New study shows potential benefits of exercise."
    source2 = "https://www.reuters.com"
    run_detector(claim2, source2)


if __name__ == "__main__":
    main()
