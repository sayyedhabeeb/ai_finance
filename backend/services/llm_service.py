from groq import Groq

from backend.config.settings import get_settings


def generate_response(user_query: str) -> str:
    settings = get_settings()
    api_key = settings.GROQ_API_KEY or ""
    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful financial AI assistant."},
            {"role": "user", "content": user_query},
        ],
    )
    content = response.choices[0].message.content or ""
    return content.strip()

