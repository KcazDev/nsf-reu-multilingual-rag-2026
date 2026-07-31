import csv
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_RESULTS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "openai_pilot_results_review.csv"
)

BM25_RESULTS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "openai_bm25_top2_review_scored.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "openai_retrieval_condition_summary.txt"
)

SUMMARY_CSV_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "openai_retrieval_condition_summary.csv"
)


def load_csv(file_path: Path) -> list[dict]:
    """Load rows from a CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n{file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def parse_score(value: str) -> int:
    """Convert a correctness score to an integer."""
    cleaned_value = str(value).strip()

    if cleaned_value not in {"0", "1", "2"}:
        raise ValueError(
            f"Invalid correctness score: {value!r}"
        )

    return int(cleaned_value)


def normalize_boolean(value: str) -> bool:
    """Convert common yes/no values into a boolean."""
    return str(value).strip().lower() in {
        "yes",
        "true",
        "1",
    }


def summarize_condition(
    rows: list[dict],
    condition_name: str,
) -> dict:
    """Calculate summary statistics for one condition."""
    condition_rows = [
        row
        for row in rows
        if str(row.get("condition", "")).strip()
        == condition_name
    ]

    if not condition_rows:
        raise ValueError(
            f"No rows were found for condition: "
            f"{condition_name}"
        )

    scores = [
        parse_score(row["correctness_score"])
        for row in condition_rows
    ]

    fully_correct = sum(
        score == 2
        for score in scores
    )

    partially_correct = sum(
        score == 1
        for score in scores
    )

    incorrect = sum(
        score == 0
        for score in scores
    )

    refusals = sum(
        normalize_boolean(
            row.get("refusal", "")
        )
        for row in condition_rows
    )

    wrong_language = sum(
        not normalize_boolean(
            row.get("correct_language", "")
        )
        for row in condition_rows
    )

    unsupported = sum(
        normalize_boolean(
            row.get(
                "unsupported_information",
                "",
            )
        )
        for row in condition_rows
    )

    total_score = sum(scores)
    possible_score = len(scores) * 2

    correctness_percentage = (
        total_score / possible_score * 100
    )

    return {
        "condition": condition_name,
        "responses": len(condition_rows),
        "total_score": total_score,
        "possible_score": possible_score,
        "correctness_percentage": (
            correctness_percentage
        ),
        "fully_correct": fully_correct,
        "partially_correct": partially_correct,
        "incorrect": incorrect,
        "refusals": refusals,
        "wrong_language": wrong_language,
        "unsupported_information": unsupported,
    }


def summarize_error_categories(
    rows: list[dict],
) -> dict[str, int]:
    """Count BM25 primary error categories."""
    counts = defaultdict(int)

    for row in rows:
        category = str(
            row.get(
                "primary_error_category",
                "",
            )
        ).strip()

        if category:
            counts[category] += 1

    return dict(
        sorted(
            counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )
    )


def main() -> None:
    original_rows = load_csv(
        ORIGINAL_RESULTS_FILE
    )

    bm25_rows = load_csv(
        BM25_RESULTS_FILE
    )

    combined_rows = (
        original_rows
        + bm25_rows
    )

    conditions = [
        "no_documents",
        "support_only",
        "all_documents",
        "bm25_top2",
    ]

    summaries = [
        summarize_condition(
            combined_rows,
            condition,
        )
        for condition in conditions
    ]

    summary_lines = [
        "GPT-5.4 RETRIEVAL CONDITION COMPARISON",
        "=" * 72,
        "",
        (
            f"{'Condition':<20}"
            f"{'Score':>10}"
            f"{'Percent':>12}"
            f"{'Full':>8}"
            f"{'Partial':>10}"
            f"{'Wrong':>8}"
            f"{'Unsupported':>14}"
        ),
        "-" * 82,
    ]

    for summary in summaries:
        score_text = (
            f"{summary['total_score']}/"
            f"{summary['possible_score']}"
        )

        summary_lines.append(
            f"{summary['condition']:<20}"
            f"{score_text:>10}"
            f"{summary['correctness_percentage']:>11.1f}%"
            f"{summary['fully_correct']:>8}"
            f"{summary['partially_correct']:>10}"
            f"{summary['incorrect']:>8}"
            f"{summary['unsupported_information']:>14}"
        )

    summary_lines.extend(
        [
            "",
            "KEY COMPARISONS",
            "=" * 72,
        ]
    )

    summary_by_condition = {
        summary["condition"]: summary
        for summary in summaries
    }

    bm25_percentage = (
        summary_by_condition[
            "bm25_top2"
        ]["correctness_percentage"]
    )

    support_percentage = (
        summary_by_condition[
            "support_only"
        ]["correctness_percentage"]
    )

    all_documents_percentage = (
        summary_by_condition[
            "all_documents"
        ]["correctness_percentage"]
    )

    no_documents_percentage = (
        summary_by_condition[
            "no_documents"
        ]["correctness_percentage"]
    )

    summary_lines.extend(
        [
            (
                "BM25 Top-2 versus no documents: "
                f"{bm25_percentage - no_documents_percentage:+.1f} "
                "percentage points"
            ),
            (
                "BM25 Top-2 versus all documents: "
                f"{bm25_percentage - all_documents_percentage:+.1f} "
                "percentage points"
            ),
            (
                "BM25 Top-2 versus oracle support only: "
                f"{bm25_percentage - support_percentage:+.1f} "
                "percentage points"
            ),
        ]
    )

    bm25_error_categories = (
        summarize_error_categories(
            bm25_rows
        )
    )

    summary_lines.extend(
        [
            "",
            "BM25 TOP-2 ERROR CATEGORIES",
            "=" * 72,
        ]
    )

    for category, count in (
        bm25_error_categories.items()
    ):
        summary_lines.append(
            f"{category}: {count}"
        )

    summary_lines.extend(
        [
            "",
            "RESEARCH INTERPRETATION",
            "=" * 72,
            (
                "BM25 Top-2 produced the highest GPT-5.4 "
                "correctness score in this 20-question pilot."
            ),
            (
                "The filtered BM25 condition outperformed the "
                "all-document condition, supporting the conclusion "
                "that removing distractor evidence can improve "
                "answer generation."
            ),
            (
                "The result does not prove that BM25 is superior to "
                "oracle retrieval because the sample is small and "
                "GPT-5.4 sometimes answered correctly even when BM25 "
                "did not retrieve all XRAG-labeled support documents."
            ),
        ]
    )

    summary_text = "\n".join(
        summary_lines
    )

    SUMMARY_FILE.write_text(
        summary_text,
        encoding="utf-8",
    )

    with SUMMARY_CSV_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        fieldnames = list(
            summaries[0].keys()
        )

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )

    print(summary_text)

    print("\nFiles created:")
    print(SUMMARY_FILE)
    print(SUMMARY_CSV_FILE)


if __name__ == "__main__":
    main()