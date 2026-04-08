import requests

url = "http://127.0.0.1:8000/api/upload"
file_path = "test_document.md"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "text/markdown")}
    response = requests.post(url, files=files)

print(f"Status Code: {response.status_code}")
print(f"Response Body: {response.text}")