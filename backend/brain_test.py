import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def test_llm():
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    print(f"Testing Groq with model: {model}")
    print(f"API Key: {api_key[:10]}...{api_key[-5:] if api_key else ''}")
    
    try:
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Say 'The AI is working' if you can read this."},
            ],
            model=model,
        )
        print("Response:", chat_completion.choices[0].message.content)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_llm()
