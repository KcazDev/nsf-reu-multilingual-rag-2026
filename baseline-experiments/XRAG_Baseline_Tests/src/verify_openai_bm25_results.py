import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "openai_bm25_top2_results.jsonl"
)

EXPECTED_EXAMPLES = 20
EXPECTED_CONDITION = "bm25_top2"
WORD_LIMIT = 35


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""
    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

    return records


def main() -> None:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"BM25 generation results were not found:\n{RESULTS_FILE}"
        )

    records = load_jsonl(RESULTS_FILE)

    example_numbers = [
        int(record["example_number"])
        for record in records
    ]

    unique_example_numbers = set(example_numbers)

    duplicate_examples = [
        example_number
        for example_number, count
        in Counter(example_numbers).items()
        if count > 1
    ]

    missing_examples = [
        example_number
        for example_number in range(
            1,
            EXPECTED_EXAMPLES + 1,
        )
        if example_number not in unique_example_numbers
    ]

    condition_counts = Counter(
        str(record.get("condition"))
        for record in records
    )

    language_counts = Counter(
        str(record.get("language"))
        for record in records
    )

    stop_reason_counts = Counter(
        str(record.get("stop_reason"))
        for record in records
    )

    complete_support_counts = Counter(
        bool(record.get("complete_support_at_2"))
        for record in records
    )

    over_word_limit = [
        record
        for record in records
        if int(record.get("word_count", 0)) > WORD_LIMIT
    ]

    empty_answers = [
        record
        for record in records
        if not str(record.get("model_answer", "")).strip()
    ]

    total_input_tokens = sum(
        int(record.get("input_tokens", 0))
        for record in records
    )

    total_output_tokens = sum(
        int(record.get("output_tokens", 0))
        for record in records
    )

    total_tokens = (
        total_input_tokens
        + total_output_tokens
    )

    average_retrieval_precision = sum(
        float(
            record.get(
                "retrieval_precision_at_2",
                0.0,
            )
        )
        for record in records
    ) / len(records)

    average_retrieval_recall = sum(
        float(
            record.get(
                "retrieval_recall_at_2",
                0.0,
            )
        )
        for record in records
    ) / len(records)

    print("BM25 TOP-2 OPENAI RESULT VERIFICATION")
    print("=" * 60)

    print(f"Results file: {RESULTS_FILE}")
    print(f"Total records: {len(records)}")
    print(
        f"Unique examples: "
        f"{len(unique_example_numbers)}"
    )

    print(
        f"Condition counts: "
        f"{dict(condition_counts)}"
    )

    print(
        f"Language counts: "
        f"{dict(language_counts)}"
    )

    print(
        f"Stop-reason counts: "
        f"{dict(stop_reason_counts)}"
    )

    print(
        f"Complete support retrieved: "
        f"{complete_support_counts.get(True, 0)}/"
        f"{len(records)}"
    )

    print("\nRetrieval averages:")
    print(
        f"Average Precision@2: "
        f"{average_retrieval_precision:.3f}"
    )
    print(
        f"Average Recall@2: "
        f"{average_retrieval_recall:.3f}"
    )

    print("\nToken usage:")
    print(
        f"Input tokens: "
        f"{total_input_tokens:,}"
    )
    print(
        f"Output tokens: "
        f"{total_output_tokens:,}"
    )
    print(
        f"Combined tokens: "
        f"{total_tokens:,}"
    )

    print("\nIntegrity checks:")
    print(f"Missing examples: {missing_examples}")
    print(f"Duplicate examples: {duplicate_examples}")
    print(f"Empty answers: {len(empty_answers)}")

    print(
        f"Responses over {WORD_LIMIT} words: "
        f"{len(over_word_limit)}"
    )

    for record in over_word_limit:
        print(
            f"  Example {record['example_number']} "
            f"(XRAG ID {record['xrag_id']}): "
            f"{record['word_count']} words"
        )

    valid_record_count = (
        len(records) == EXPECTED_EXAMPLES
        and len(unique_example_numbers)
        == EXPECTED_EXAMPLES
        and not missing_examples
        and not duplicate_examples
        and not empty_answers
        and condition_counts.get(
            EXPECTED_CONDITION,
            0,
        )
        == EXPECTED_EXAMPLES
    )

    print("\nFinal verification:")
    print(
        "PASS"
        if valid_record_count
        else "REVIEW REQUIRED"
    )


if __name__ == "__main__":
    main()