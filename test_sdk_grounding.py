import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
print("Sending SDK search grounding request...")
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Find 2 active Software Engineer Intern roles in Bangalore.",
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
    )
    print("Response:")
    print(response.text)
except Exception as e:
    print(f"Error: {e}")
