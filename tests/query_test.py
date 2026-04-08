import requests
import json

def test_query():
    url = "http://127.0.0.1:8000/api/query"
    payload = {
        "query": "What is the solar system?",
        "user_id": "test_user",
        "session_id": "test_session"
    }
    headers = {
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes

        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(response.json())

    except requests.exceptions.RequestException as e:
        print(f"Connection Error: {e}")
        print("Please ensure the FastAPI server is running.")

if __name__ == "__main__":
    test_query()