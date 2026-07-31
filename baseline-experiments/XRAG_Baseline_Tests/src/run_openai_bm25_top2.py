import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi


MODEL_NAME = "gpt-5.4-2026-03-05"
MAX_OUTPUT_TOKENS = 180
PROMPT_VERSION = "controlled_v2"

# Test one example before running all 20.
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
    """Load XRAG examples from a JSONL file."""
    examples = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

    return examples


def load_completed_results(
    output_path: Path,
) -> set[tuple[int, str]]:
    """Load completed example-condition pairs for safe resume."""
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
                    int(record["example_number"]),
                    str(record["condition"]),
                )
            )

    return completed


def get_article_text(article: Any) -> str:
    """Return article text from either a string or dictionary."""
    if isinstance(article, str):
        return article

    if isinstance(article, dict):
        possible_fields = [
            "title",
            "text",
            "content",
            "article",
            "body",
        ]

        text_parts = []

        for field in possible_fields:
            value = article.get(field)

            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())

        return " ".join(text_parts)

    return str(article)


def tokenize(text: str) -> list[str]:
    """
    Convert Arabic or Spanish text into lowercase Unicode word tokens.

    This matches the simple tokenizer used in the BM25 baseline.
    """
    return re.findall(
        r"\w+",
        text.lower(),
        flags=re.UNICODE,
    )


def rank_articles_with_bm25(
    question: str,
    articles: list[Any],
) -> list[tuple[int, float]]:
    """
    Rank all candidate articles using BM25.

    Returns zero-based article indexes and their BM25 scores.
    """
    if not articles:
        raise ValueError(
            "The XRAG example does not contain any articles."
        )

    tokenized_articles = [
        tokenize(get_article_text(article))
        for article in articles
    ]

    tokenized_question = tokenize(question)

    if not tokenized_question:
        raise ValueError(
            "The question did not produce any BM25 tokens."
        )

    bm25 = BM25Okapi(tokenized_articles)

    scores = bm25.get_scores(
        tokenized_question
    )

    ranked_articles = sorted(
        enumerate(scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        (
            article_index,
            float(score),
        )
        for article_index, score in ranked_articles
    ]


def get_supporting_indexes(
    support_labels: list[Any],
) -> list[int]:
    """Return zero-based XRAG supporting-article indexes."""
    supporting_indexes = []

    for article_index, label in enumerate(
        support_labels
    ):
        try:
            is_support = float(label) > 0
        except (TypeError, ValueError):
            is_support = False

        if is_support:
            supporting_indexes.append(
                article_index
            )

    return supporting_indexes


def calculate_retrieval_metrics(
    selected_indexes: list[int],
    supporting_indexes: list[int],
) -> tuple[float, float, bool]:
    """Calculate top-two precision, recall, and full coverage."""
    supporting_articles_found = [
        article_index
        for article_index in selected_indexes
        if article_index in supporting_indexes
    ]

    precision_at_2 = (
        len(supporting_articles_found)
        / len(selected_indexes)
        if selected_indexes
        else 0.0
    )

    recall_at_2 = (
        len(supporting_articles_found)
        / len(supporting_indexes)
        if supporting_indexes
        else 0.0
    )

    complete_support = set(
        supporting_indexes
    ).issubset(
        set(selected_indexes)
    )

    return (
        precision_at_2,
        recall_at_2,
        complete_support,
    )


def extract_answer(response: Any) -> str:
    """Extract generated text from the OpenAI response."""
    return (
        response.output_text or ""
    ).strip()


def get_stop_reason(response: Any) -> str:
    """Return a readable response completion reason."""
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
    """
    Format the BM25-selected articles for the model.

    Documents are presented in BM25 ranking order.
    """
    document_sections = []

    for document_number, article_index in enumerate(
        article_indexes,
        start=1,
    ):
        article_text = get_article_text(
            example["articles"][article_index]
        )

        article_date = (
            example["dates"][article_index]
        )

        article_url = (
            example["urls"][article_index]
        )

        section = (
            f"DOCUMENT {document_number}\n"
            f"Original article number: {article_index + 1}\n"
            f"Publication date: {article_date}\n"
            f"Source URL: {article_url}\n"
            f"Article text:\n{article_text}"
        )

        document_sections.append(section)

    return "\n\n".join(
        document_sections
    )


def build_user_prompt(
    question: str,
    documents_text: str,
) -> str:
    """Create the user prompt containing retrieved documents."""
    return (
        f"RETRIEVED DOCUMENTS:\n\n"
        f"{documents_text}\n\n"
        f"QUESTION:\n{question}"
    )


def save_result(
    output_path: Path,
    record: dict[str, Any],
) -> None:
    """Append one completed result to the JSONL output file."""
    with output_path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def main() -> None:
    project_folder = (
        Path(__file__).resolve().parents[1]
    )

    env_path = (
        project_folder
        / ".env"
    )

    input_path = (
        project_folder
        / "data"
        / "xrag_pilot_sample.jsonl"
    )

    output_path = (
        project_folder
        / "outputs"
        / "openai_bm25_top2_results.jsonl"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Pilot dataset was not found:\n{input_path}"
        )

    load_dotenv(env_path)

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY was not found. "
            "Check the .env file."
        )

    client = OpenAI(
        api_key=api_key
    )

    examples = load_examples(
        input_path
    )

    examples = examples[
        :MAX_EXAMPLES
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed_results = (
        load_completed_results(
            output_path
        )
    )

    total_planned_calls = len(
        examples
    )

    completed_this_run = 0
    skipped_this_run = 0

    print("XRAG OPENAI BM25 TOP-2 TEST")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print("Retriever: BM25Okapi")
    print("Condition: bm25_top2")
    print(
        f"Examples included: "
        f"{len(examples)}"
    )
    print(
        f"Maximum planned calls: "
        f"{total_planned_calls}"
    )
    print(
        f"Output file: "
        f"{output_path}"
    )

    for example_number, example in enumerate(
        examples,
        start=1,
    ):
        condition_name = "bm25_top2"

        result_key = (
            example_number,
            condition_name,
        )

        print("\n" + "#" * 70)
        print(
            f"EXAMPLE {example_number} "
            f"OF {len(examples)}"
        )
        print(
            f"XRAG ID: "
            f"{example['id']}"
        )
        print(
            f"Language: "
            f"{example['language']}"
        )
        print(
            f"Task: "
            f"{example['task']}"
        )

        question = example["question"]

        print(
            f"Question: "
            f"{question}"
        )

        if result_key in completed_results:
            print(
                "\nSkipping completed condition: "
                f"{condition_name}"
            )

            skipped_this_run += 1
            continue

        articles = example["articles"]

        ranked_articles = (
            rank_articles_with_bm25(
                question=question,
                articles=articles,
            )
        )

        bm25_top2_indexes = [
            article_index
            for article_index, score
            in ranked_articles[:2]
        ]

        bm25_ranking_one_based = [
            article_index + 1
            for article_index, score
            in ranked_articles
        ]

        bm25_scores_by_rank = [
            round(score, 6)
            for article_index, score
            in ranked_articles
        ]

        supporting_indexes = (
            get_supporting_indexes(
                example["is_support"]
            )
        )

        (
            retrieval_precision_at_2,
            retrieval_recall_at_2,
            complete_support_at_2,
        ) = calculate_retrieval_metrics(
            selected_indexes=(
                bm25_top2_indexes
            ),
            supporting_indexes=(
                supporting_indexes
            ),
        )

        documents_text = format_documents(
            example=example,
            article_indexes=(
                bm25_top2_indexes
            ),
        )

        user_prompt = build_user_prompt(
            question=question,
            documents_text=documents_text,
        )

        print("\nBM25 RETRIEVAL")
        print("-" * 70)
        print(
            "Full ranking: "
            f"{bm25_ranking_one_based}"
        )
        print(
            "Selected top two articles: "
            f"{[index + 1 for index in bm25_top2_indexes]}"
        )
        print(
            "True supporting articles: "
            f"{[index + 1 for index in supporting_indexes]}"
        )
        print(
            f"Retrieval Precision@2: "
            f"{retrieval_precision_at_2:.2f}"
        )
        print(
            f"Retrieval Recall@2: "
            f"{retrieval_recall_at_2:.2f}"
        )
        print(
            "Complete support retrieved: "
            f"{complete_support_at_2}"
        )

        print("\nOPENAI GENERATION")
        print("-" * 70)
        print(
            f"Running condition: "
            f"{condition_name}"
        )
        print(
            f"Documents used: "
            f"{len(bm25_top2_indexes)}"
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
                max_output_tokens=(
                    MAX_OUTPUT_TOKENS
                ),
                store=False,
            )

            model_answer = extract_answer(
                response
            )

            word_count = len(
                model_answer.split()
            )

            stop_reason = get_stop_reason(
                response
            )

            input_tokens = (
                response.usage.input_tokens
            )

            output_tokens = (
                response.usage.output_tokens
            )

            record = {
                "timestamp_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
                "example_number": (
                    example_number
                ),
                "xrag_id": example["id"],
                "language": example["language"],
                "task": example["task"],
                "question_type": example["q_type"],
                "answer_type": example["a_type"],
                "crossdoc_type": (
                    example["crossdoc_type"]
                ),
                "question": question,
                "gold_answer": example["answer"],
                "model": MODEL_NAME,
                "prompt_version": PROMPT_VERSION,
                "condition": condition_name,
                "retriever": "BM25Okapi",
                "retriever_tokenizer": (
                    "unicode_lowercase_words"
                ),
                "selected_article_indexes_zero_based": (
                    bm25_top2_indexes
                ),
                "selected_article_numbers_one_based": [
                    index + 1
                    for index in bm25_top2_indexes
                ],
                "bm25_full_ranking_one_based": (
                    bm25_ranking_one_based
                ),
                "bm25_scores_by_rank": (
                    bm25_scores_by_rank
                ),
                "true_support_article_numbers_one_based": [
                    index + 1
                    for index in supporting_indexes
                ],
                "retrieval_precision_at_2": (
                    retrieval_precision_at_2
                ),
                "retrieval_recall_at_2": (
                    retrieval_recall_at_2
                ),
                "complete_support_at_2": (
                    complete_support_at_2
                ),
                "model_answer": model_answer,
                "word_count": word_count,
                "stop_reason": stop_reason,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

            save_result(
                output_path,
                record,
            )

            completed_results.add(
                result_key
            )

            completed_this_run += 1

            print("\nOPENAI ANSWER:")
            print(model_answer)

            print("\nOUTPUT CHECK:")
            print(
                f"Word count: "
                f"{word_count}"
            )
            print(
                f"Stop reason: "
                f"{stop_reason}"
            )

            print("\nTOKEN USAGE:")
            print(
                f"Input tokens: "
                f"{input_tokens}"
            )
            print(
                f"Output tokens: "
                f"{output_tokens}"
            )

        except Exception as error:
            print("\nERROR:")
            print(
                f"{type(error).__name__}: "
                f"{error}"
            )
            print(
                "Completed results were already saved. "
                "Run the program again to resume."
            )

        time.sleep(
            PAUSE_BETWEEN_CALLS_SECONDS
        )

    print("\n" + "=" * 70)
    print(
        "ONE-EXAMPLE BM25 TOP-2 "
        "OPENAI TEST FINISHED"
    )
    print(
        f"New calls completed: "
        f"{completed_this_run}"
    )
    print(
        "Previously completed calls skipped: "
        f"{skipped_this_run}"
    )
    print("RESULTS SAVED TO:")
    print(output_path)


if __name__ == "__main__":
    main()