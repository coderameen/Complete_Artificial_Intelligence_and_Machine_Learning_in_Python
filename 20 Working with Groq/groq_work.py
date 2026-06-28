import os
from groq import Groq
from groq import APIError

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    api_key = input("Enter your GROQ API key: ").strip()
    if not api_key:
        raise RuntimeError(
            "A GROQ API key is required. Set GROQ_API_KEY or enter it when prompted."
        )

user_prompt = input("Enter the text you want the model to respond to: ").strip()
if not user_prompt:
    raise RuntimeError("No input text provided. Please enter a prompt and run again.")

client = Groq(api_key=api_key)
try:
    completion = client.chat.completions.create(
        model="groq/compound",
        messages=[
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=1,
        max_completion_tokens=256,
        top_p=1,
        stream=False,
        stop=None,
    )
except APIError as exc:
    raise RuntimeError(
        f"Groq API request failed: {exc}. "
        "Try a smaller prompt or check your model/settings."
    ) from exc

print(completion.choices[0].message.content)
