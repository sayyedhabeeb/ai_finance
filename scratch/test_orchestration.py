import requests
import json
import time

def test_query(query):
    url = "http://localhost:8000/api/v1/query"
    payload = {"query": query}
    headers = {"Content-Type": "application/json"}
    
    print(f"\n--- Testing Query: '{query}' ---")
    t0 = time.perf_counter()
    try:
        response = requests.post(url, json=payload, headers=headers)
        elapsed = time.perf_counter() - t0
        
        if response.status_code == 200:
            data = response.json()
            print(f"Status: SUCCESS ({elapsed:.2f}s)")
            print(f"Agents Used: {data.get('agents_used')}")
            print(f"Confidence: {data.get('confidence')}")
            print(f"Answer: {data.get('answer')[:200]}...")
            if data.get('sources'):
                print(f"Sources: {data.get('sources')}")
        else:
            print(f"Status: FAILED ({response.status_code})")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test 1: General query
    test_query("Hello, who are you?")
    
    # Test 2: Portfolio query (should trigger PortfolioManager)
    test_query("Analyze my current portfolio holdings")
    
    # Test 3: Multi-agent query (should trigger Portfolio + Risk)
    test_query("What is the risk profile of my portfolio and should I rebalance?")
