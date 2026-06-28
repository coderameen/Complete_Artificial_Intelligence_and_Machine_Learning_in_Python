import requests

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": "Bearer YOUR_GROQ_API",
    "Content-Type": "application/json"
}

payload = {
    "model": "llama-3.1-8b-instant",
    "messages": [
        {
            "role": "user",
            "content": "what is AI"
        }
    ]
}

response = requests.post(
    url,
    headers=headers,
    json=payload
)

print(response.status_code)
print(response.json()['content'])