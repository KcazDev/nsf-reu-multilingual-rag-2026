import csv
from collections import Counter, defaultdict
from pathlib import Path


CONDITION_ORDER = [
    "no_documents",
    "support_only",
    "all_documents",
]


def normalize_text(value: str) -> str:
    return value.strip().lower()


def main() -> None:
    project_folder = Path(__file__).resolve().parents[1]

    input_path = (
        project_folder
        / "outputs"
        / "pilot_results_review.csv"
    )

    output_path = (
        project_folder
        / "outputs"
        / "pilot_scoring_summary.txt"
    )

    records = []

    with input_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row_number, row in enumerate(reader, start=2):
            score_text = row["correctness_score"].strip()

            if score_text == "":
                raise ValueError(
                    f"Missing correctness score on Excel row {row_number}."
                )

            try:
                score = int(score_text)
            except ValueError:
                raise ValueError(
                    f"Invalid correctness score on Excel row "
                    f"{row_number}: {score_text}"
                )

            if score not in {0, 1, 2}:
                raise ValueError(
                    f"Score must be 0, 1, or 2 on Excel row "
                    f"{row_number}."
                )

            records.append(
                {
                    "condition": row["condition"].strip(),
                    "language": row["language"].strip(),
                    "task": row["task"].strip(),
                    "question_type": row["question_type"].strip(),
                    "crossdoc_type": row["crossdoc_type"].strip(),
                    "score": score,
                    "refusal": normalize_text(row["refusal"]),
                    "correct_language": normalize_text(
                        row["correct_language"]
                    ),
                    "unsupported_information": normalize_text(
                        row["unsupported_information"]
                    ),
                }
            )

    results_by_condition = defaultdict(list)
    results_by_condition_language = defaultdict(list)

    for record in records:
        results_by_condition[
            record["condition"]
        ].append(record)

        results_by_condition_language[
            (
                record["condition"],
                record["language"],
            )
        ].append(record)

    report_lines = []

    report_lines.append("XRAG PILOT SCORING SUMMARY")
    report_lines.append("=" * 75)
    report_lines.append("")
    report_lines.append(
        f"Total scored responses: {len(records)}"
    )
    report_lines.append("")

    report_lines.append("RESULTS BY RETRIEVAL CONDITION")
    report_lines.append("-" * 75)

    for condition in CONDITION_ORDER:
        condition_records = results_by_condition[condition]

        if not condition_records:
            continue

        scores = [
            record["score"]
            for record in condition_records
        ]

        score_counts = Counter(scores)

        average_score = sum(scores) / len(scores)
        accuracy_percentage = (
            sum(scores)
            / (len(scores) * 2)
            * 100
        )

        refusal_count = sum(
            record["refusal"] == "yes"
            for record in condition_records
        )

        correct_language_count = sum(
            record["correct_language"] == "yes"
            for record in condition_records
        )

        unsupported_count = sum(
            record["unsupported_information"] == "yes"
            for record in condition_records
        )

        report_lines.append("")
        report_lines.append(f"Condition: {condition}")
        report_lines.append(
            f"  Responses: {len(condition_records)}"
        )
        report_lines.append(
            f"  Average correctness score: "
            f"{average_score:.2f} out of 2"
        )
        report_lines.append(
            f"  Percentage of possible points: "
            f"{accuracy_percentage:.1f}%"
        )
        report_lines.append(
            f"  Fully correct scores (2): "
            f"{score_counts[2]}"
        )
        report_lines.append(
            f"  Partially correct scores (1): "
            f"{score_counts[1]}"
        )
        report_lines.append(
            f"  Incorrect scores (0): "
            f"{score_counts[0]}"
        )
        report_lines.append(
            f"  Refusals: "
            f"{refusal_count} "
            f"({refusal_count / len(condition_records) * 100:.1f}%)"
        )
        report_lines.append(
            f"  Correct-language responses: "
            f"{correct_language_count} "
            f"({correct_language_count / len(condition_records) * 100:.1f}%)"
        )
        report_lines.append(
            f"  Responses with unsupported information: "
            f"{unsupported_count} "
            f"({unsupported_count / len(condition_records) * 100:.1f}%)"
        )

    report_lines.append("")
    report_lines.append("")
    report_lines.append(
        "RESULTS BY CONDITION AND LANGUAGE"
    )
    report_lines.append("-" * 75)

    for condition in CONDITION_ORDER:
        for language in ["ar", "es"]:
            group = results_by_condition_language[
                (condition, language)
            ]

            if not group:
                continue

            scores = [
                record["score"]
                for record in group
            ]

            average_score = sum(scores) / len(scores)
            point_percentage = (
                sum(scores)
                / (len(scores) * 2)
                * 100
            )

            report_lines.append(
                f"{condition:15} | "
                f"Language: {language} | "
                f"Responses: {len(group):2} | "
                f"Average: {average_score:.2f}/2 | "
                f"Points: {point_percentage:.1f}%"
            )

    report_lines.append("")
    report_lines.append("")
    report_lines.append(
        "BEST CONDITION"
    )
    report_lines.append("-" * 75)

    condition_averages = {}

    for condition in CONDITION_ORDER:
        group = results_by_condition[condition]

        if group:
            condition_averages[condition] = (
                sum(record["score"] for record in group)
                / len(group)
            )

    best_condition = max(
        condition_averages,
        key=condition_averages.get,
    )

    report_lines.append(
        f"Highest average score: {best_condition} "
        f"with {condition_averages[best_condition]:.2f} out of 2."
    )

    report_text = "\n".join(report_lines)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.write(report_text)

    print(report_text)

    print("")
    print("=" * 75)
    print("SUMMARY SAVED TO:")
    print(output_path)


if __name__ == "__main__":
    main()