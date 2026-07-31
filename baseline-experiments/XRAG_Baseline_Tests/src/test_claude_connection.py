from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv


def main() -> None:
    project_folder = Path(__file__).resolve().parents[1]
    env_path = project_folder / ".env"

    load_dotenv(env_path)

    client = Anthropic()

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=50,
        messages=[
            {
                "role": "user",
                "content": (
                    "Reply with exactly this sentence: "
                    "Claude API connection successful."
                ),
            }
        ],
    )

    print(response.content[0].text)
    print(f"Input tokens: {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")


if __name__ == "__main__":
    main()