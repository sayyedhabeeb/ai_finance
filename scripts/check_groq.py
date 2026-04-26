import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_groq():
    """Standalone script to verify Groq API connectivity and key validity."""
    # Load .env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        print(f"Loading environment from: {env_path}")
        load_dotenv(env_path)
    else:
        print("Warning: .env file not found in project root.")

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    if not api_key:
        print("ERROR: GROQ_API_KEY not found in environment or .env file.")
        return

    print(f"Using Model: {model}")
    print(f"Using API Key: {api_key[:6]}...{api_key[-4:]}")

    try:
        client = Groq(api_key=api_key)
        print("\nSending test query to Groq...")
        
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Please confirm you are working by saying: 'Groq is active and responding.'"}
            ],
            temperature=0.5,
            max_tokens=100
        )
        
        response = completion.choices[0].message.content
        print("\n--- Response ---")
        print(response)
        print("----------------\n")
        
        if "Groq is active" in response or "responding" in response.lower():
            print("SUCCESS: Groq API is working perfectly!")
        else:
            print("INFO: Received a response, but it didn't match the expected confirmation text.")
            
    except Exception as e:
        print(f"\nERROR: Failed to connect to Groq API.")
        print(f"Details: {str(e)}")
        if "401" in str(e) or "invalid_api_key" in str(e):
            print("TIP: Your API key seems to be invalid. Please check your GROQ_API_KEY in .env")

if __name__ == "__main__":
    check_groq()
