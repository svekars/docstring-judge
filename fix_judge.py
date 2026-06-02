import os
import json
from google import genai

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not set")
    exit(1)

client = genai.Client(api_key=api_key)

# The model 'gemini-1.5-flash' doesn't seem to be in the list, let's use 'gemini-2.0-flash'
model_name = "gemini-2.0-flash"

try:
    print(f"Testing generation with {model_name}...")
    response = client.models.generate_content(
        model=model_name,
        contents="Hello, world!"
    )
    print(response.text)
except Exception as e:
    print(f"Error: {e}")
