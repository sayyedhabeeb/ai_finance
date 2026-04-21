import os
import sys
from dotenv import load_dotenv

# Load .env first
load_dotenv()

# Add project root to path
sys.path.append(os.getcwd())

from backend.services.llm_service import generate_response

print("Testing live Groq integration via backend service...")
try:
    response = generate_response("What is the current status of the Groq integration?")
    print("\n[SUCCESS] Groq Response:")
    print(response)
except Exception as e:
    print(f"\n[FAILED] Error: {e}")
