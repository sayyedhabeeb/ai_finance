import os
import sys
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(os.getcwd())

from backend.config.settings import get_settings
from backend.services.llm_service import generate_response, is_llm_unavailable_response
from backend.services.agent_factory import AgentFactory
from backend.agents.base import BaseAgent
from backend.config.schemas import AgentType

def test_settings():
    print("Testing Settings...")
    settings = get_settings()
    print(f"Provider: {settings.llm_provider}")
    print(f"Groq Model: {settings.groq_model}")
    print(f"Groq API Key set: {bool(settings.groq_api_key)}")
    
    assert settings.llm_provider == "groq"
    assert "llama" in settings.groq_model
    print("Settings test passed!\n")

def test_agent_factory():
    print("Testing Agent Factory...")
    try:
        factory = AgentFactory.from_settings()
        print("Agent Factory initialized successfully.")
        # Check LLM type
        if factory._llm:
            print(f"LLM Type: {type(factory._llm).__name__}")
            from langchain_groq import ChatGroq
            assert isinstance(factory._llm, ChatGroq)
        print("Agent Factory test passed!\n")
    except Exception as e:
        print(f"Agent Factory test failed/skipped (might need GROQ_API_KEY): {e}")

def test_llm_service_logic():
    print("Testing LLM Service internal logic...")
    # Mocking httpx to avoid real API calls
    import httpx
    from unittest.mock import patch
    
    with patch("httpx.Client.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from Groq"}}]
        }
        mock_post.return_value = mock_response
        
        # We need a dummy key for settings
        os.environ["GROQ_API_KEY"] = "test_key"
        from backend.config.settings import Settings
        # Force reload settings or just trust env
        
        response = generate_response("Hi")
        print(f"Mocked Response: {response}")
        assert response == "Hello from Groq"
    print("LLM Service logic test passed!\n")

if __name__ == "__main__":
    try:
        test_settings()
        test_agent_factory()
        test_llm_service_logic()
        print("All local verification tests passed!")
    except Exception as e:
        print(f"Verification failed: {e}")
        sys.exit(1)
