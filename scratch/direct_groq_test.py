import os
from groq import Groq
from dotenv import load_dotenv

# Load .env
load_dotenv()

key = os.getenv("GROQ_API_KEY")
print(f"Key being tested: {key[:10]}...")

client = Groq(api_key=key)

try:
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "test"}]
    )
    print("Success! Response:")
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"FAILED: {e}")
