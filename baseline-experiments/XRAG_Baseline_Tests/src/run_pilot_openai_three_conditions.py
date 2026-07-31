import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


MODEL_NAME = "gpt-5.4-2026-03-05"
MAX_OUTPUT_TOKENS = 180
PROMPT_VERSION = "controlled_v2"

# Test only one example first.
# After verification, change this to 20.
MAX_EXAMPLES = 20

PAUSE_BETWEEN_CALLS_SECONDS = 1


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


def load_examples(file_path: Path) -> list[dict[str, Any]]:
    examples = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                examples.append(json.loads(line))

    return examples


def load_completed_results(
    output_path: Path,
) -> set[tuple[int, str]]:
    completed = set()

    if not output_path.exists():
        return completed

    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            completed.add(
                (
                    record["example_number"],
                    record["condition"],
                )
            )

    return completed


def extract_answer(response: Any) -> str:
    return (response.output_text or "").strip()


def get_stop_reason(response: Any) -> str:
    if response.status == "completed":
        return "end_turn"

    if response.status == "incomplete":
        incomplete_details = getattr(
            response,
            "incomplete_details",
            None,
        )

        reason = getattr(
            incomplete_details,
            "reason",
            None,
        )

        return reason or "incomplete"

    return response.status or "unknown"


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


def save_result(
    output_path: Path,
    record: dict[str, Any],
) -> None:
    with output_path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
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
        / "openai_pilot_three_conditions_results.jsonl"
    )

    load_dotenv(env_path)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY was not found. Check the .env file."
        )

    client = OpenAI(api_key=api_key)

    examples = load_examples(input_path)
    examples = examples[:MAX_EXAMPLES]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed_results = load_completed_results(output_path)

    total_planned_calls = len(examples) * 3
    completed_this_run = 0
    skipped_this_run = 0

    print("XRAG OPENAI PILOT BATCH TEST")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Examples included: {len(examples)}")
    print("Planned conditions per example: 3")
    print(f"Maximum planned calls: {total_planned_calls}")
    print(f"Output file: {output_path}")

    for example_number, example in enumerate(
        examples,
        start=1,
    ):
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

        print("\n" + "#" * 70)
        print(
            f"EXAMPLE {example_number} OF {len(examples)}"
        )
        print(f"XRAG ID: {example['id']}")
        print(f"Language: {example['language']}")
        print(f"Task: {example['task']}")
        print(f"Question: {question}")

        for condition_data in conditions:
            condition_name = condition_data["condition"]

            result_key = (
                example_number,
                condition_name,
            )

            if result_key in completed_results:
                print(
                    f"\nSkipping completed condition: "
                    f"{condition_name}"
                )

                skipped_this_run += 1
                continue

            user_prompt = build_user_prompt(
                question,
                condition_data["documents_text"],
            )

            print("\n" + "-" * 70)
            print(f"Running condition: {condition_name}")
            print(
                f"Documents used: "
                f"{len(condition_data['article_indexes'])}"
            )

            try:
                response = client.responses.create(
                    model=MODEL_NAME,
                    instructions=SYSTEM_PROMPT,
                    input=user_prompt,
                    reasoning={
                        "effort": "none",
                    },
                    temperature=0,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    store=False,
                )

                model_answer = extract_answer(response)
                word_count = len(model_answer.split())
                stop_reason = get_stop_reason(response)

                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

                record = {
                    "timestamp_utc": (
                        datetime.now(timezone.utc).isoformat()
                    ),
                    "example_number": example_number,
                    "xrag_id": example["id"],
                    "language": example["language"],
                    "task": example["task"],
                    "question_type": example["q_type"],
                    "answer_type": example["a_type"],
                    "crossdoc_type": example["crossdoc_type"],
                    "question": question,
                    "gold_answer": example["answer"],
                    "model": MODEL_NAME,
                    "prompt_version": PROMPT_VERSION,
                    "condition": condition_name,
                    "article_indexes": (
                        condition_data["article_indexes"]
                    ),
                    "model_answer": model_answer,
                    "word_count": word_count,
                    "stop_reason": stop_reason,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }

                save_result(output_path, record)

                completed_results.add(result_key)
                completed_this_run += 1

                print("\nOPENAI ANSWER:")
                print(model_answer)

                print("\nOUTPUT CHECK:")
                print(f"Word count: {word_count}")
                print(f"Stop reason: {stop_reason}")

                print("\nTOKEN USAGE:")
                print(f"Input tokens: {input_tokens}")
                print(f"Output tokens: {output_tokens}")

            except Exception as error:
                print("\nERROR:")
                print(
                    f"{type(error).__name__}: {error}"
                )
                print(
                    "Completed results were already saved. "
                    "Run the program again to resume."
                )

            time.sleep(PAUSE_BETWEEN_CALLS_SECONDS)

    print("\n" + "=" * 70)
    print("ONE-EXAMPLE OPENAI TEST FINISHED")
    print(f"New calls completed: {completed_this_run}")
    print(f"Previously completed calls skipped: {skipped_this_run}")
    print("RESULTS SAVED TO:")
    print(output_path)


if __name__ == "__main__":
    main()