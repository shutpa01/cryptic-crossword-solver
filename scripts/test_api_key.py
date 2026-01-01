import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Force-load .env from project root
project_root = Path(__file__).resolve().parents[1]
env_path = project_root / ".env"

print("Looking for .env at:", env_path)

load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")
print("API Key loaded? ", bool(api_key))

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Check .env formatting.")

# Try a small request
client = OpenAI(api_key=api_key)

print("Trying test request...")

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hello"}],
    max_tokens=5
)

print(resp.choices[0].message.content)
