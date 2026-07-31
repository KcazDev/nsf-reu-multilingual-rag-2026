import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "openai_bm25_top2_results.jsonl"
)

REVIEW_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "openai_bm25_top2_review.csv"
)

WORD_LIMIT = 35


def load_jsonl(
    file_path: Path,
) -> list[dict[str, Any]]:
    """Load all records from a JSONL file."""
    records = []

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {error}"
                ) from error

    return records


def format_list(
    values: list[Any] | None,
) -> str:
    """Convert a list into a readable comma-separated string."""
    if not values:
        return ""

    return ",".join(
        str(value)
        for value in values
    )


def main() -> None:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            "BM25 Top-2 results were not found:\n"
            f"{RESULTS_FILE}"
        )

    records = load_jsonl(
        RESULTS_FILE
    )

    if not records:
        raise ValueError(
            "The BM25 Top-2 results file is empty."
        )

    records.sort(
        key=lambda record: int(
            record["example_number"]
        )
    )

    review_rows = []

    for record in records:
        word_count = int(
            record.get(
                "word_count",
                0,
            )
        )

        review_rows.append(
            {
                "example_number": (
                    record.get(
                        "example_number"
                    )
                ),
                "xrag_id": (
                    record.get(
                        "xrag_id"
                    )
                ),
                "language": (
                    record.get(
                        "language"
                    )
                ),
                "task": (
                    record.get(
                        "task"
                    )
                ),
                "question_type": (
                    record.get(
                        "question_type"
                    )
                ),
                "crossdoc_type": (
                    record.get(
                        "crossdoc_type"
                    )
                ),
                "condition": (
                    record.get(
                        "condition"
                    )
                ),
                "question": (
                    record.get(
                        "question"
                    )
                ),
                "gold_answer": (
                    record.get(
                        "gold_answer"
                    )
                ),
                "model_answer": (
                    record.get(
                        "model_answer"
                    )
                ),
                "selected_articles": format_list(
                    record.get(
                        "selected_article_numbers_one_based"
                    )
                ),
                "true_support_articles": format_list(
                    record.get(
                        "true_support_article_numbers_one_based"
                    )
                ),
                "retrieval_precision_at_2": (
                    record.get(
                        "retrieval_precision_at_2"
                    )
                ),
                "retrieval_recall_at_2": (
                    record.get(
                        "retrieval_recall_at_2"
                    )
                ),
                "complete_support_at_2": (
                    record.get(
                        "complete_support_at_2"
                    )
                ),
                "word_count": word_count,
                "over_35_words": (
                    "yes"
                    if word_count > WORD_LIMIT
                    else "no"
                ),
                "correctness_score": "",
                "refusal": "",
                "correct_language": "",
                "unsupported_information": "",
                "primary_error_category": "",
                "review_notes": "",
                "input_tokens": (
                    record.get(
                        "input_tokens"
                    )
                ),
                "output_tokens": (
                    record.get(
                        "output_tokens"
                    )
                ),
            }
        )

    REVIEW_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        review_rows[0].keys()
    )

    with REVIEW_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            review_rows
        )

    print(
        "BM25 TOP-2 REVIEW SHEET CREATED"
    )
    print("=" * 60)

    print(
        f"Responses included: "
        f"{len(review_rows)}"
    )

    print(
        f"Review file: "
        f"{REVIEW_FILE}"
    )

    print(
        "\nManual scoring columns added:"
    )

    print(
        "- correctness_score"
    )
    print(
        "- refusal"
    )
    print(
        "- correct_language"
    )
    print(
        "- unsupported_information"
    )
    print(
        "- primary_error_category"
    )
    print(
        "- review_notes"
    )


if __name__ == "__main__":
    main()