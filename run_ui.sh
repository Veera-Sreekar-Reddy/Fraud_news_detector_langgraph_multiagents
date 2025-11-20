#!/bin/bash
# Start the Streamlit UI
# This script ensures the correct Python environment is used

echo "=========================================="
echo "🔍 Multi-Agent Disinformation Detector UI"
echo "=========================================="
echo ""

# Check if langgraph is installed
echo "Checking dependencies..."
python -c "import langgraph" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ langgraph not found. Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

# Check if streamlit is installed
python -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ streamlit not found. Installing..."
    pip install streamlit
    echo ""
fi

echo "✓ Dependencies verified"
echo ""
echo "Starting Streamlit UI..."
echo "The UI will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""
echo "=========================================="
echo ""

# Run streamlit with explicit Python
python -m streamlit run app.py

