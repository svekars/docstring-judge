import os
from google import genai
import google.genai.errors

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not set")
    exit(1)

client = genai.Client(api_key=api_key)
try:
    print("Attempting to list models...")
    models = client.models.list()
    for m in models:
        print(f"Model: {m.name}")
except Exception as e:
    print(f"Error: {e}")
    print(f"Type: {type(e)}")
