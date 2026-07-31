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
        / "example_1_support_only.json"
    )

    load_dotenv(env_path)

    with input_path.open("r", encoding="utf-8") as file:
        first_example = json.loads(file.readline())

    question = first_example["question"]
    article_order = first_example["article_order"]
    support_labels = first_example["is_support"]

    supporting_indexes = [
        article_index
        for article_index in article_order
        if support_labels[article_index] > 0
    ]

    document_sections = []

    for document_number, article_index in enumerate(
        supporting_indexes,
        start=1,
    ):
        article_text = first_example["articles"][article_index]
        article_date = first_example["dates"][article_index]
        article_url = first_example["urls"][article_index]
        support_label = support_labels[article_index]

        document_section = (
            f"DOCUMENT {document_number}\n"
            f"Stored article index: {article_index}\n"
            f"Publication date: {article_date}\n"
            f"URL: {article_url}\n"
            f"Text:\n{article_text}\n"
        )

        document_sections.append(document_section)

    documents_text = "\n\n".join(document_sections)

    system_prompt = (
        "You are completing a multilingual factual question-answering "
        "benchmark using retrieved news documents. "
        "Use only the information contained in the supplied documents. "
        "This is an academic evaluation of factual news understanding, "
        "not a request for operational advice or recommendations. "
        "Answer directly and concisely in the same language as the question. "
        "Do not explain your reasoning. "
        "Do not add information that is not supported by the documents."
    )

    user_prompt = (
        f"RETRIEVED DOCUMENTS:\n\n{documents_text}\n\n"
        f"QUESTION:\n{question}"
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
        "prompt_version": "support_only_v1",
        "condition": "support_only",
        "xrag_id": first_example["id"],
        "language": first_example["language"],
        "task": first_example["task"],
        "question_type": first_example["q_type"],
        "crossdoc_type": first_example["crossdoc_type"],
        "question": question,
        "gold_answer": first_example["answer"],
        "model_answer": model_answer,
        "article_order": article_order,
        "supporting_article_indexes": supporting_indexes,
        "supporting_article_labels": [
            support_labels[index]
            for index in supporting_indexes
        ],
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

    print("SUPPORT-ONLY TEST COMPLETE")
    print("=" * 60)

    print(f"XRAG ID: {first_example['id']}")
    print(f"Language: {first_example['language']}")
    print(f"Model: {MODEL_NAME}")
    print(f"Supporting documents used: {len(supporting_indexes)}")
    print(f"Supporting article indexes: {supporting_indexes}")

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