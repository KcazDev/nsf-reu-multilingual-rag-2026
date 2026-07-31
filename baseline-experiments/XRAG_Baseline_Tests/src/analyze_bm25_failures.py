import csv
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "bm25_retrieval_results.csv"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "bm25_failure_analysis.txt"
)


def parse_integer_list(value: str) -> list[int]:
    """Convert a comma-separated string into a list of integers."""
    if not value or not value.strip():
        return []

    return [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def string_to_bool(value: str) -> bool:
    """Convert a CSV boolean value into a Python boolean."""
    return value.strip().lower() == "true"


def load_results(file_path: Path) -> list[dict]:
    """Load the BM25 CSV results."""
    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def calculate_support_ranks(
    full_ranking: list[int],
    true_support_articles: list[int],
) -> list[int]:
    """Return the rank position of every supporting article."""
    support_ranks = []

    for support_article in true_support_articles:
        if support_article in full_ranking:
            support_ranks.append(
                full_ranking.index(support_article) + 1
            )

    return support_ranks


def add_group_result(
    group_data: dict,
    group_name: str,
    row: dict,
) -> None:
    """Add one example to a grouped summary."""
    group_data[group_name]["examples"] += 1
    group_data[group_name]["recall_at_2"] += float(
        row["recall_at_2"]
    )
    group_data[group_name]["reciprocal_rank"] += float(
        row["reciprocal_rank"]
    )

    if string_to_bool(row["all_support_at_2"]):
        group_data[group_name]["complete_at_2"] += 1


def format_group_summary(
    title: str,
    group_data: dict,
) -> list[str]:
    """Create readable summary lines for one grouping."""
    lines = [
        "",
        title,
        "-" * len(title),
    ]

    for group_name in sorted(group_data):
        values = group_data[group_name]
        examples = values["examples"]

        average_recall = (
            values["recall_at_2"] / examples
        )

        average_mrr = (
            values["reciprocal_rank"] / examples
        )

        lines.append(
            f"{group_name}: "
            f"examples={examples}, "
            f"complete_support_at_2="
            f"{values['complete_at_2']}/{examples}, "
            f"average_recall_at_2={average_recall:.3f}, "
            f"average_MRR={average_mrr:.3f}"
        )

    return lines


def main() -> None:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"BM25 results file was not found:\n{RESULTS_FILE}"
        )

    rows = load_results(RESULTS_FILE)

    language_summary = defaultdict(
        lambda: {
            "examples": 0,
            "complete_at_2": 0,
            "recall_at_2": 0.0,
            "reciprocal_rank": 0.0,
        }
    )

    task_summary = defaultdict(
        lambda: {
            "examples": 0,
            "complete_at_2": 0,
            "recall_at_2": 0.0,
            "reciprocal_rank": 0.0,
        }
    )

    question_type_summary = defaultdict(
        lambda: {
            "examples": 0,
            "complete_at_2": 0,
            "recall_at_2": 0.0,
            "reciprocal_rank": 0.0,
        }
    )

    report_lines = [
        "BM25 FAILURE ANALYSIS",
        "=" * 60,
        f"Examples analyzed: {len(rows)}",
        "",
        "QUESTIONS WHERE BM25 MISSED COMPLETE SUPPORT IN TOP 2",
        "=" * 60,
    ]

    failed_examples = 0
    supporting_articles_at_rank_3 = 0

    for row in rows:
        add_group_result(
            language_summary,
            row["language"],
            row,
        )

        add_group_result(
            task_summary,
            row["task"],
            row,
        )

        add_group_result(
            question_type_summary,
            row["question_type"],
            row,
        )

        full_ranking = parse_integer_list(
            row["bm25_full_ranking"]
        )

        true_support_articles = parse_integer_list(
            row["true_support_articles"]
        )

        support_ranks = calculate_support_ranks(
            full_ranking,
            true_support_articles,
        )

        if len(full_ranking) >= 3:
            third_article = full_ranking[2]

            if third_article in true_support_articles:
                supporting_articles_at_rank_3 += 1

        if not string_to_bool(row["all_support_at_2"]):
            failed_examples += 1

            report_lines.extend(
                [
                    "",
                    f"Example {row['example_number']}",
                    f"XRAG ID: {row['xrag_id']}",
                    f"Language: {row['language']}",
                    f"Task: {row['task']}",
                    f"Question type: {row['question_type']}",
                    f"Question: {row['question']}",
                    (
                        "True supporting articles: "
                        f"{true_support_articles}"
                    ),
                    (
                        "BM25 ranking: "
                        f"{full_ranking}"
                    ),
                    (
                        "Supporting-document ranks: "
                        f"{support_ranks}"
                    ),
                    (
                        "Recall@2: "
                        f"{float(row['recall_at_2']):.2f}"
                    ),
                    "-" * 60,
                ]
            )

    report_lines.extend(
        [
            "",
            "OVERALL FAILURE SUMMARY",
            "=" * 60,
            (
                "Questions missing complete support in top 2: "
                f"{failed_examples}/{len(rows)}"
            ),
            (
                "Supporting articles appearing exactly at rank 3: "
                f"{supporting_articles_at_rank_3}"
            ),
        ]
    )

    report_lines.extend(
        format_group_summary(
            "PERFORMANCE BY LANGUAGE",
            language_summary,
        )
    )

    report_lines.extend(
        format_group_summary(
            "PERFORMANCE BY RETRIEVAL TASK",
            task_summary,
        )
    )

    report_lines.extend(
        format_group_summary(
            "PERFORMANCE BY QUESTION TYPE",
            question_type_summary,
        )
    )

    report_text = "\n".join(report_lines)

    REPORT_FILE.write_text(
        report_text,
        encoding="utf-8",
    )

    print(report_text)
    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()