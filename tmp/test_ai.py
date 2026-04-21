import requests
import json

def test_api():
    url = "http://localhost:8000/api/v1/query"
    payload = {"query": "What is 2+2? Only answer with the number."}
    headers = {"Content-Type": "application/json"}
    
    print(f"Testing API at {url}...")
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
