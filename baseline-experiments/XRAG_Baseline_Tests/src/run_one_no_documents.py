import json
from datetime import datetime, timezone
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv


MODEL_NAME = "claude-haiku-4-5-20251001"


def main() -> None:
    project_folder = Path(__file__).resolve().parents[1]

    env_path = project_folder / ".env"
    input_path = project_folder / "data" / "xrag_pilot_sample.jsonl"
    output_path = (
        project_folder
        / "outputs"
        / "example_1_no_documents.json"
    )

    load_dotenv(env_path)

    with input_path.open("r", encoding="utf-8") as file:
        first_example = json.loads(file.readline())

    question = first_example["question"]

    system_prompt = (
        "You are participating in a multilingual question-answering "
        "evaluation. Answer the question directly and concisely. "
        "Answer in the same language as the question. "
        "Do not explain your reasoning."
    )

    user_prompt = (
        "No retrieved documents are available.\n\n"
        f"Question:\n{question}"
    )

    client = Anthropic()

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=300,
        temperature=0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    )

    model_answer = "".join(
        block.text
        for block in response.content
        if block.type == "text"
    ).strip()

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "condition": "no_documents",
        "xrag_id": first_example["id"],
        "language": first_example["language"],
        "task": first_example["task"],
        "question_type": first_example["q_type"],
        "crossdoc_type": first_example["crossdoc_type"],
        "question": question,
        "gold_answer": first_example["answer"],
        "model_answer": model_answer,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("SINGLE NO-DOCUMENT TEST COMPLETE")
    print("=" * 60)
    print(f"XRAG ID: {first_example['id']}")
    print(f"Language: {first_example['language']}")
    print(f"Model: {MODEL_NAME}")

    print("\nQUESTION:")
    print(question)

    print("\nGOLD ANSWER:")
    print(first_example["answer"])

    print("\nCLAUDE ANSWER:")
    print(model_answer)

    print("\nTOKEN USAGE:")
    print(f"Input tokens: {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")

    print("\nRESULT SAVED TO:")
    print(output_path)


if __name__ == "__main__":
    main()