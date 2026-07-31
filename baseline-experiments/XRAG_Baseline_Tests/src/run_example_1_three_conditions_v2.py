import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv


MODEL_NAME = "claude-haiku-4-5-20251001"
MAX_OUTPUT_TOKENS = 180
PROMPT_VERSION = "controlled_v2"


SYSTEM_PROMPT = (
    "You are completing a multilingual factual question-answering benchmark "
    "based on retrospective public news reporting. Your task is to report "
    "what the news says, not to provide advice or operational instructions. "
    "Return exactly one sentence in the same language as the question. "
    "Use no more than 35 words. Do not use headings, bullet points, caveats, "
    "recommendations, refusals, or explanations. "
    "When retrieved documents are supplied, use the documents most directly "
    "relevant to the question and respect the time frame implied by the "
    "question. Ignore irrelevant documents and later events that do not "
    "answer the original question. When no documents are supplied, answer "
    "using your existing general knowledge."
)


def extract_answer(response: Any) -> str:
    text_blocks = []

    for block in response.content:
        if getattr(block, "type", None) == "text":
            text_blocks.append(block.text)

    return "".join(text_blocks).strip()


def format_documents(
    example: dict[str, Any],
    article_indexes: list[int],
) -> str:
    document_sections = []

    for document_number, article_index in enumerate(
        article_indexes,
        start=1,
    ):
        article_text = example["articles"][article_index]
        article_date = example["dates"][article_index]
        article_url = example["urls"][article_index]

        section = (
            f"DOCUMENT {document_number}\n"
            f"Publication date: {article_date}\n"
            f"Source URL: {article_url}\n"
            f"Article text:\n{article_text}"
        )

        document_sections.append(section)

    return "\n\n".join(document_sections)


def build_user_prompt(
    question: str,
    documents_text: str | None,
) -> str:
    if documents_text is None:
        return f"QUESTION:\n{question}"

    return (
        f"RETRIEVED DOCUMENTS:\n\n"
        f"{documents_text}\n\n"
        f"QUESTION:\n{question}"
    )


def main() -> None:
    project_folder = Path(__file__).resolve().parents[1]

    env_path = project_folder / ".env"

    input_path = (
        project_folder
        / "data"
        / "xrag_pilot_sample.jsonl"
    )

    output_path = (
        project_folder
        / "outputs"
        / "example_1_three_conditions_v2.json"
    )

    load_dotenv(env_path)

    with input_path.open("r", encoding="utf-8") as file:
        example = json.loads(file.readline())

    question = example["question"]
    article_order = example["article_order"]
    support_labels = example["is_support"]

    supporting_indexes = [
        article_index
        for article_index in article_order
        if support_labels[article_index] > 0
    ]

    all_article_indexes = list(article_order)

    conditions = [
        {
            "condition": "no_documents",
            "article_indexes": [],
            "documents_text": None,
        },
        {
            "condition": "support_only",
            "article_indexes": supporting_indexes,
            "documents_text": format_documents(
                example,
                supporting_indexes,
            ),
        },
        {
            "condition": "all_documents",
            "article_indexes": all_article_indexes,
            "documents_text": format_documents(
                example,
                all_article_indexes,
            ),
        },
    ]

    client = Anthropic()
    results = []

    print("CONTROLLED THREE-CONDITION TEST V2")
    print("=" * 70)
    print(f"XRAG ID: {example['id']}")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt version: {PROMPT_VERSION}")

    for condition_data in conditions:
        condition_name = condition_data["condition"]

        user_prompt = build_user_prompt(
            question,
            condition_data["documents_text"],
        )

        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        )

        model_answer = extract_answer(response)
        word_count = len(model_answer.split())

        result = {
            "condition": condition_name,
            "article_indexes": condition_data["article_indexes"],
            "model_answer": model_answer,
            "word_count": word_count,
            "stop_reason": response.stop_reason,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        results.append(result)

        print("\n" + "-" * 70)
        print(f"CONDITION: {condition_name}")
        print(
            f"Documents used: "
            f"{len(condition_data['article_indexes'])}"
        )

        print("\nCLAUDE ANSWER:")
        print(model_answer)

        print("\nOUTPUT CHECK:")
        print(f"Word count: {word_count}")
        print(f"Stop reason: {response.stop_reason}")

        print("\nTOKEN USAGE:")
        print(f"Input tokens: {response.usage.input_tokens}")
        print(f"Output tokens: {response.usage.output_tokens}")

    complete_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "xrag_id": example["id"],
        "language": example["language"],
        "task": example["task"],
        "question_type": example["q_type"],
        "crossdoc_type": example["crossdoc_type"],
        "question": question,
        "gold_answer": example["answer"],
        "article_order": article_order,
        "support_labels": support_labels,
        "supporting_article_indexes": supporting_indexes,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            complete_record,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("ALL THREE CONDITIONS COMPLETE")
    print("RESULT SAVED TO:")
    print(output_path)


if __name__ == "__main__":
    main()