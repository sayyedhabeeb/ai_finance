import logging
import os
from groq import Groq
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

def _get_groq_client() -> Groq:
    """Initialize and return a Groq client using app settings or environment."""
    settings = get_settings()
    api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY is missing or not loaded from environment")
    
    return Groq(api_key=api_key)


def _fallback_response(user_query: str, reason: str) -> str:
    return (
        "AI provider (Groq) is temporarily unavailable, so I cannot generate a full model-based answer right now.\n\n"
        f"Reason: {reason}\n\n"
        f"Your query: {user_query}\n\n"
        "Please retry in a moment. Check your GROQ_API_KEY if this persists."
    )


def is_llm_unavailable_response(text: str) -> bool:
    lowered = (text or "").lower()
    markers = (
        "ai provider (groq) is temporarily unavailable",
        "failed to connect to groq",
        "groq api returned status",
        "groq response was empty",
    )
    return any(marker in lowered for marker in markers)


def generate_response(user_query: str, system_prompt: str = "You are a financial AI assistant.") -> str:
    """Generate a response using dynamic Groq client initialization."""
    settings = get_settings()
    model = settings.groq_model or "llama-3.1-8b-instant"

    api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY missing or not loaded during query")
        return _fallback_response(user_query, "GROQ_API_KEY is not configured in environment.")

    try:
        # Dynamic initialization to ensure latest env var is used
        client = _get_groq_client()
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            model=model,
            temperature=0.7,
            max_tokens=1024,
        )

        content = chat_completion.choices[0].message.content
        if not content:
            return _fallback_response(user_query, "Groq response was empty.")
        return str(content).strip()
    except Exception as exc:
        logger.error("Error calling Groq API: %s", str(exc))
        return _fallback_response(user_query, f"Failed to connect to Groq ({exc}).")
