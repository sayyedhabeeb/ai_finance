import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.llm_service import generate_response
from backend.config.settings import get_settings

def test_llm():
    settings = get_settings()
    print(f"Testing LLM Provider: {settings.llm_provider}")
    print(f"Model: {settings.groq_model}")
    
    try:
        response = generate_response("Hello, are you working?", system_prompt="Test prompt")
        print("LLM Response:")
        print(response)
    except Exception as e:
        print(f"LLM Test failed: {e}")

if __name__ == "__main__":
    test_llm()
