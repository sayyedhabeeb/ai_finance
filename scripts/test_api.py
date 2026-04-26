import requests
import json

def query_backend():
    url = "http://localhost:8000/api/v1/query"
    payload = {
        "query": "hello",
        "user_id": "test",
        "session_id": "test"
    }
    try:
        response = requests.post(url, json=payload)
        print("Status Code:", response.status_code)
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    query_backend()
