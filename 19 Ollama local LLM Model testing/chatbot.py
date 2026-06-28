import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

payload = {
    "model": "llama3.2:1b",
    "prompt": "What is Python in a single line?",
    "stream": False
}

response = requests.post(OLLAMA_URL, json=payload)

if response.status_code == 200:
    result = response.json()
    print("Response:")
    print(result["response"])
else:
    print(f"Error: {response.status_code}")
    print(response.text)