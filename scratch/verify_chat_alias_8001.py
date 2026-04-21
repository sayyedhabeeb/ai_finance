import requests
import json

def test_chat_alias():
    url = "http://localhost:8001/chat"
    payload = {"message": "Hello, this is a test of the chat alias on port 8001."}
    headers = {"Content-Type": "application/json"}
    
    print(f"Testing Chat Alias at {url}...")
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response Data:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error Response: {response.text}")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_chat_alias()
