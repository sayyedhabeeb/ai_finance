import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.orchestrator import Orchestrator

async def test_orchestrator():
    print("Initialising Orchestrator...")
    try:
        # Pass critic_enabled=False to simplify first test
        orchestrator = Orchestrator(critic_enabled=False)
        print("Orchestrator initialised.")
        
        query = "Hello, what is the market outlook?"
        print(f"Processing query: '{query}'")
        
        # Test with a timeout
        task = asyncio.create_task(orchestrator.process_query(query))
        result = await asyncio.wait_for(task, timeout=30.0)
        
        print("\nOrchestrator Result:")
        print(result.get("response") or result.get("answer"))
    except asyncio.TimeoutError:
        print("\nERROR: Orchestrator timed out after 30s!")
    except Exception as e:
        print(f"\nERROR: Orchestrator failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_orchestrator())
