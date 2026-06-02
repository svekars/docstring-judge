import os
import google.generativeai as genai

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not set")
    exit(1)

genai.configure(api_key=api_key)

print("Listing models available to your API key:")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f" - {m.name}")
except Exception as e:
    print(f"Failed to list models: {e}")

print("\nTesting gemini-1.5-flash content generation:")
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Hello, respond with 'OK'")
    print(f"Success! Response: {response.text}")
except Exception as e:
    print(f"Failed to generate content: {e}")
