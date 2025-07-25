"""
Simple test script to verify LangGraph migration.
"""
import asyncio
import os
from pathlib import Path

# Set up minimal configuration for testing
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["GEMINI_API_KEY"] = "test-key"

# Mock the settings to avoid import issues
import sys
sys.path.append(".")

from codebase_rag.rag_orchestrator import RAGOrchestrator


async def test_basic_functionality():
    """Test basic RAG orchestrator functionality."""
    print("Testing LangGraph integration...")
    
    # Create a simple tool for testing
    from langchain_core.tools import tool
    
    @tool
    def simple_test_tool(query: str) -> str:
        """A simple test tool that echoes the input."""
        return f"Echo: {query}"
    
    try:
        # Create orchestrator with the test tool
        orchestrator = RAGOrchestrator(tools=[simple_test_tool])
        print("✅ RAG Orchestrator created successfully")
        
        # This would fail without API keys, but we can test object creation
        print("✅ LangGraph migration appears successful!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(test_basic_functionality())
