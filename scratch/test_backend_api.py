import requests

def test_query():
    url = "http://localhost:8000/api/v1/query"
    payload = {"query": "hello"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_query()
