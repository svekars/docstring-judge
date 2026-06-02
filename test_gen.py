import os

from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
try:
    print(f"Trying gemini-3.5-flash...")
    response = client.models.generate_content(model="gemini-3.5-flash", contents="Hi")
    print("Success")
except Exception as e:
    print(f"Error: {e}")
