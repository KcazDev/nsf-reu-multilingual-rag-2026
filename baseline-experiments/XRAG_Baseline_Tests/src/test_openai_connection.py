import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY was not found. Check the .env file."
    )

client = OpenAI(api_key=api_key)

response = client.responses.create(
    model="gpt-5.4-2026-03-05",
    input="Reply with exactly: OpenAI connection successful",
    reasoning={"effort": "none"},
    max_output_tokens=40,
    store=False,
)

print("OPENAI RESPONSE:")
print(response.output_text)

print("\nTOKEN USAGE:")
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")